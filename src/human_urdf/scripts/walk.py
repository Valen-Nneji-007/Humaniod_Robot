#!/usr/bin/env python3
"""
walk_gait_node.py
==================

Thin ROS2 wrapper around gait_generator.py. Publishes a repeating
JointTrajectory to your `body_controller`, one full gait cycle at a
time, re-sending the next cycle just before the current one finishes
so the walk continues indefinitely.

`body_controller` covers your WHOLE robot's 9 commanded joints
(confirmed via human_controllers.yaml): shoulders, elbows, neck, and
the 4 leg joints together, in one JointTrajectoryController with
`allow_partial_joints_goal: false`. So every message this node sends
includes all 9 joints: the 4 leg joints follow the gait from
gait_generator.py, everything else is held at a fixed neutral
position (see NEUTRAL_POSE below).

--------------------------------------------------------------------
STARTUP SEQUENCE (added after a first real test toppled the robot)
--------------------------------------------------------------------
A first test showed the robot falling over almost immediately after
this node started. The most likely trigger: the previous version
jumped straight from the robot's resting pose (all joints at 0) to
the gait's full swing amplitude in just 0.1s -- an abrupt snap, not a
step, with zero lateral balance margin to absorb it. This version
fixes that with a three-phase startup, all sent as ONE trajectory
message so the controller has the whole plan up front:
    1. SETTLE (`settle_time`, default 2s): hold the all-zero rest pose,
       so the robot is confirmed physically stable before anything moves
    2. RAMP (`ramp_time`, default 3s): smoothly ease (not jump) from
       rest into the gait cycle's own starting pose
    3. WALK: the normal repeating gait cycle, from here on identical to
       before
This does NOT fix the underlying lack of hip-roll/ankle -- see the
docstring in gait_generator.py. If the robot still falls even with
this gentle startup and the slower defaults now in GaitParams, that's
strong evidence the fall is the fundamental lateral-stability issue,
not a startup transient -- worth first testing in isolation, e.g. by
setting hip_swing_amp / knee_lift to ~0 (this should then just make it
stand still) and seeing whether it stays upright doing NOTHING before
blaming the gait itself.

--------------------------------------------------------------------
BEFORE RUNNING -- confirm your actual controller name/topic
--------------------------------------------------------------------
    ros2 control list_controllers

If your controller isn't named `body_controller`, either:
  (a) change TRAJ_TOPIC below to match, e.g.
      "/leg_controller/joint_trajectory", or
  (b) pass it at launch: --ros-args -p traj_topic:=/your/topic

If your controller's joint list differs from ALL_JOINTS below, update
it to match human_controllers.yaml's `body_controller.joints` exactly.

--------------------------------------------------------------------
RUNNING
--------------------------------------------------------------------
    ros2 run <your_package> walk.py

Stop it (Ctrl+C) and the last trajectory segment will simply finish
executing -- there's no "stop" trajectory sent automatically.
"""

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from gait_generator import GaitParams, JOINT_NAMES, sample_cycle, cycle_start_pose

TRAJ_TOPIC = "/body_controller/joint_trajectory"

# Exact joint list `body_controller` commands, taken directly from
# human_controllers.yaml (src/human_controller/config/human_controllers.yaml).
# Grippers are NOT part of this controller, and `allow_partial_joints_goal:
# false` means every message must contain exactly these 9 joints, no more,
# no less.
ALL_JOINTS = [
    "joint_shoulderR", "joint_elbowR", "joint_shoulderL", "joint_elbowL",
    "joint_neck", "joint_waistR", "joint_kneeR", "joint_waistL", "joint_kneeL",
]

# Fixed position (rad) for every joint this gait doesn't animate.
NEUTRAL_POSE = {name: 0.0 for name in ALL_JOINTS if name not in JOINT_NAMES}


def _smoothstep(x):
    x = max(0.0, min(1.0, x))
    return x * x * (3 - 2 * x)


class WalkGaitNode(Node):
    def __init__(self):
        super().__init__("walk_gait_node")

        # Expose every gait parameter via ROS2 params so you can tune live with
        # `ros2 param set` instead of editing/rebuilding, e.g.:
        #   ros2 param set /walk_gait_node cycle_period 6.0
        self.declare_parameter("traj_topic", TRAJ_TOPIC)
        self.declare_parameter("cycle_period", 9.0)
        self.declare_parameter("duty_factor", 0.88)
        self.declare_parameter("hip_swing_amp", 0.14)
        self.declare_parameter("knee_neutral", 0.10)
        self.declare_parameter("knee_lift_R", 0.20)
        self.declare_parameter("knee_lift_L", 0.08)
        self.declare_parameter("samples_per_cycle", 40)
        self.declare_parameter("settle_time", 2.0)
        self.declare_parameter("ramp_time", 3.0)
        # how long before a cycle ends to send the next one, so the
        # controller always has a queued segment (avoids any gap/stall)
        self.declare_parameter("resend_lead_time", 0.3)

        topic = self.get_parameter("traj_topic").value
        self.pub = self.create_publisher(JointTrajectory, topic, 10)
        self.get_logger().info(f"Publishing walk gait to: {topic}")
        self.get_logger().info(f"Full joint set ({len(ALL_JOINTS)}): {ALL_JOINTS}")

        self._build_params()
        self._settle_time = self.get_parameter("settle_time").value
        self._ramp_time = self.get_parameter("ramp_time").value

        startup_len = self._settle_time + self._ramp_time
        self._publish_startup_then_cycle()

        resend_period = max(0.5, self.params.cycle_period - self.get_parameter("resend_lead_time").value)
        # first repeating cycle only starts once settle+ramp has finished playing out
        self.timer = self.create_timer(startup_len + resend_period, self._publish_cycle)
        self.get_logger().info(
            f"Startup: {self._settle_time:.1f}s settle, then {self._ramp_time:.1f}s ramp into gait, "
            f"then repeating {self.params.cycle_period:.1f}s walk cycles."
        )

    def _build_params(self):
        self.params = GaitParams(
            cycle_period=self.get_parameter("cycle_period").value,
            duty_factor=self.get_parameter("duty_factor").value,
            hip_swing_amp=self.get_parameter("hip_swing_amp").value,
            knee_neutral=self.get_parameter("knee_neutral").value,
            knee_lift_R=self.get_parameter("knee_lift_R").value,
            knee_lift_L=self.get_parameter("knee_lift_L").value,
        )
        self.n_samples = self.get_parameter("samples_per_cycle").value

    def _full_positions(self, leg_pose):
        """leg_pose: dict with the 4 leg joint names -> angle. Returns a
        position list covering ALL_JOINTS, everything else at NEUTRAL_POSE."""
        row = dict(NEUTRAL_POSE)
        row.update(leg_pose)
        return [row[name] for name in ALL_JOINTS]

    def _publish_startup_then_cycle(self):
        """One combined trajectory: hold rest -> smoothly ramp into the gait's
        start pose -> play one full gait cycle -> (the timer takes over from
        here, repeating _publish_cycle)."""
        msg = JointTrajectory()
        msg.joint_names = list(ALL_JOINTS)
        points = []

        rest_pose = {name: 0.0 for name in JOINT_NAMES}
        target_pose = cycle_start_pose(self.params)

        # 1) settle: hold rest pose
        pt = JointTrajectoryPoint()
        pt.positions = self._full_positions(rest_pose)
        pt.time_from_start = Duration(seconds=0.1).to_msg()
        points.append(pt)
        pt = JointTrajectoryPoint()
        pt.positions = self._full_positions(rest_pose)
        pt.time_from_start = Duration(seconds=self._settle_time).to_msg()
        points.append(pt)

        # 2) ramp: smoothstep from rest to the gait's own phase-0 pose
        n_ramp = 20
        for i in range(1, n_ramp + 1):
            s = _smoothstep(i / n_ramp)
            pose = {name: rest_pose[name] + s * (target_pose[name] - rest_pose[name]) for name in JOINT_NAMES}
            pt = JointTrajectoryPoint()
            pt.positions = self._full_positions(pose)
            pt.time_from_start = Duration(seconds=self._settle_time + s * self._ramp_time).to_msg()
            points.append(pt)

        # 3) one full gait cycle, time-shifted after settle+ramp. Skip i=0:
        #    the ramp above already lands exactly on the cycle's phase-0 pose
        #    at t=t0, so including it again would give two points at the same
        #    timestamp (t0), which JointTrajectoryController rejects as
        #    "not strictly increasing".
        t0 = self._settle_time + self._ramp_time
        t, angles = sample_cycle(self.params, n_samples=self.n_samples)
        for i in range(1, self.n_samples):
            pose = {name: float(angles[name][i]) for name in JOINT_NAMES}
            pt = JointTrajectoryPoint()
            pt.positions = self._full_positions(pose)
            pt.time_from_start = Duration(seconds=t0 + float(t[i])).to_msg()
            points.append(pt)

        msg.points = points
        self.pub.publish(msg)
        self.get_logger().info(f"Published startup sequence ({len(points)} points).")

    def _publish_cycle(self):
        t, angles = sample_cycle(self.params, n_samples=self.n_samples)

        msg = JointTrajectory()
        msg.joint_names = list(ALL_JOINTS)
        start_delay = 0.1
        for i in range(self.n_samples):
            pose = {name: float(angles[name][i]) for name in JOINT_NAMES}
            pt = JointTrajectoryPoint()
            pt.positions = self._full_positions(pose)
            pt.time_from_start = Duration(seconds=float(t[i]) + start_delay).to_msg()
            msg.points.append(pt)
        pose0 = {name: float(angles[name][0]) for name in JOINT_NAMES}
        pt_end = JointTrajectoryPoint()
        pt_end.positions = self._full_positions(pose0)
        pt_end.time_from_start = Duration(seconds=self.params.cycle_period + start_delay).to_msg()
        msg.points.append(pt_end)

        self.pub.publish(msg)
        self.get_logger().debug(f"Published gait cycle ({self.n_samples + 1} points, {self.params.cycle_period:.1f}s)")


def main():
    rclpy.init()
    node = WalkGaitNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()