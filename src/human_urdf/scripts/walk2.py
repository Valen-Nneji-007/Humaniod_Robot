#!/usr/bin/env python3
"""
walk_gait_node.py
==================

Thin ROS2 wrapper around gait_generator.py. Publishes a repeating
JointTrajectory to your `body_controller`, one full gait cycle at a
time, re-sending the next cycle just before the current one finishes
so the walk continues indefinitely.

`body_controller` covers your WHOLE robot (confirmed via
`ros2 topic echo /joint_states --once`) -- arms, neck, grippers, and
legs together -- and JointTrajectoryController rejects any message
that doesn't name every one of its configured joints. So every
message this node sends includes all 13 joints: the 4 leg joints
follow the gait from gait_generator.py, and everything else
(shoulders, elbows, neck, grippers) is held at a fixed neutral
position (see NEUTRAL_POSE below -- edit those if you want the arms
resting somewhere other than straight down / grippers somewhere other
than closed).

--------------------------------------------------------------------
BEFORE RUNNING -- confirm your actual controller name/topic
--------------------------------------------------------------------
    ros2 control list_controllers

If your controller isn't named `body_controller`, either:
  (a) change TRAJ_TOPIC below to match, e.g.
      "/leg_controller/joint_trajectory", or
  (b) pass it at launch: --ros-args -p traj_topic:=/your/topic

If your controller's joint list differs from ALL_JOINTS below (e.g.
you split it into separate arm/leg controllers later), update
ALL_JOINTS and NEUTRAL_POSE to match -- run
`ros2 topic echo /joint_states --once` again and copy the `name:`
list exactly, including any `_mimic` suffixes.

--------------------------------------------------------------------
RUNNING
--------------------------------------------------------------------
    ros2 run <your_package> walk_gait_node.py
    # or directly, if this file is executable and ROS2 is sourced:
    ./walk_gait_node.py

Stop it (Ctrl+C) and the last trajectory segment will simply finish
executing -- there's no "stop" trajectory sent automatically. Add one
if you want it to freeze in place immediately instead.
"""

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from gait_generator import GaitParams, JOINT_NAMES, sample_cycle

TRAJ_TOPIC = "/body_controller/joint_trajectory"

# Exact joint list `body_controller` commands, taken directly from
# human_controllers.yaml (src/human_controller/config/human_controllers.yaml).
# Confirmed: grippers are NOT part of this controller at all (handled
# elsewhere), and `allow_partial_joints_goal: false` means every message
# must contain exactly these 9 joints, no more, no less.
ALL_JOINTS = [
    "joint_shoulderR", "joint_elbowR", "joint_shoulderL", "joint_elbowL",
    "joint_neck", "joint_waistR", "joint_kneeR", "joint_waistL", "joint_kneeL",
]

# Fixed position (rad) for every joint this gait doesn't animate. All zeros
# is a safe default -- within every joint's <limit> range in your URDF --
# but feel free to change these (e.g. a small elbow bend) once walking works.
NEUTRAL_POSE = {name: 0.0 for name in ALL_JOINTS if name not in JOINT_NAMES}


class WalkGaitNode(Node):
    def __init__(self):
        super().__init__("walk_gait_node")

        # Expose every gait parameter via ROS2 params so you can tune live with
        # `ros2 param set` instead of editing/rebuilding, e.g.:
        #   ros2 param set /walk_gait_node cycle_period 6.0
        self.declare_parameter("traj_topic", TRAJ_TOPIC)
        self.declare_parameter("cycle_period", 4.0)
        self.declare_parameter("duty_factor", 0.8)
        self.declare_parameter("hip_swing_amp", 0.28)
        self.declare_parameter("knee_neutral", 0.12)
        self.declare_parameter("knee_lift_R", 0.30)
        self.declare_parameter("knee_lift_L", 0.06)
        self.declare_parameter("samples_per_cycle", 40)
        # how long before a cycle ends to send the next one, so the
        # controller always has a queued segment (avoids any gap/stall)
        self.declare_parameter("resend_lead_time", 0.3)

        topic = self.get_parameter("traj_topic").value
        self.pub = self.create_publisher(JointTrajectory, topic, 10)
        self.get_logger().info(f"Publishing walk gait to: {topic}")
        self.get_logger().info(f"Full joint set ({len(ALL_JOINTS)}): {ALL_JOINTS}")

        self._build_params()
        self._publish_cycle()

        resend_period = max(0.5, self.params.cycle_period - self.get_parameter("resend_lead_time").value)
        self.timer = self.create_timer(resend_period, self._publish_cycle)

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

    def _full_positions(self, angles, i):
        """Build a position list covering ALL_JOINTS: leg joints from the
        gait sample at index i, everything else held at NEUTRAL_POSE."""
        row = dict(NEUTRAL_POSE)
        for name in JOINT_NAMES:
            row[name] = float(angles[name][i])
        return [row[name] for name in ALL_JOINTS]

    def _publish_cycle(self):
        t, angles = sample_cycle(self.params, n_samples=self.n_samples)

        msg = JointTrajectory()
        msg.joint_names = list(ALL_JOINTS)
        # small lead time so the controller has a moment to receive/queue
        # this message before the first point is due
        start_delay = 0.1
        for i in range(self.n_samples):
            pt = JointTrajectoryPoint()
            pt.positions = self._full_positions(angles, i)
            pt.time_from_start = Duration(seconds=float(t[i]) + start_delay).to_msg()
            msg.points.append(pt)
        # repeat the first point at the end (closes the loop cleanly)
        pt_end = JointTrajectoryPoint()
        pt_end.positions = self._full_positions(angles, 0)
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