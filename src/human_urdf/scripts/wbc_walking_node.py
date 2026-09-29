#!/usr/bin/env python3
"""
ROS 2 node integrating walking pattern generation with WBC.

Subscribes to robot state, publishes joint torque/position commands.
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import numpy as np
import threading
import time
from std_srvs.srv import Trigger

# Import your modules
from walking_pattern import WalkingPatternGenerator, WalkParams
from wbc_controller import WholeBodyController, WBCConfig


class WBCWalkingNode(Node):
    """
    Whole-Body Control Walking Node.

    Lifecycle:
      1. Generate walking pattern (footsteps, CoM trajectory)
      2. At each control tick: solve WBC QP → send joint commands
      3. Smooth transition to standing at end of walk
    """

    # Joint ordering must match your URDF ros2_control config
    JOINT_NAMES = [
        "joint_waistR",    # 0: right hip (pitch)
        "joint_kneeR",     # 1: right knee
        "joint_waistL",    # 2: left hip (pitch)
        "joint_kneeL",     # 3: left knee
        "joint_shoulderR", # 4: right shoulder
        "joint_elbowR",    # 5: right elbow
        "joint_shoulderL", # 6: left shoulder
        "joint_elbowL",    # 7: left elbow
    ]
    N_JOINTS = len(JOINT_NAMES)

    def __init__(self):
        super().__init__('wbc_walking_node')

        # ---- Parameters ----
        self.declare_parameter('n_steps', 10)
        self.declare_parameter('step_length', 0.12)
        self.declare_parameter('step_duration', 0.8)
        self.declare_parameter('com_height', 0.8)
        self.declare_parameter('control_rate', 100.0)  # Hz

        # ---- WBC Config ----
        wbc_cfg = WBCConfig(
            n_joints=self.N_JOINTS,
            dt=1.0 / self.get_parameter('control_rate').value,
        )
        # Joint limits from URDF
        wbc_cfg.q_min = np.array([-1.57, -1.57, -1.57, -0.2,
                                   -1.57, -0.875, -1.57, -0.942])
        wbc_cfg.q_max = np.array([1.57, 0.75, 1.57, 1.57,
                                   1.57, 1.57, 1.57, 1.57])
        self.wbc = WholeBodyController(wbc_cfg)

        # ---- Walking Pattern Generator ----
        walk_params = WalkParams(
            com_height=self.get_parameter('com_height').value,
            step_length=self.get_parameter('step_length').value,
            step_duration=self.get_parameter('step_duration').value,
        )
        self.wpg = WalkingPatternGenerator(walk_params)

        # ---- State variables ----
        self.q = np.zeros(self.N_JOINTS)
        self.q_dot = np.zeros(self.N_JOINTS)
        self.state_lock = threading.Lock()

        # Walking pattern cache
        self.pattern = None
        self.pattern_idx = 0
        self.is_walking = False
        self.walk_start_time = 0.0

        # ---- ROS Interface ----
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self._joint_state_cb, 10)

        # Option A: torque commands (effort_controllers)
        self.torque_pub = self.create_publisher(
            Float64MultiArray, '/effort_commands', 10)

        # Option B: position commands (position_controllers)
        self.position_pub = self.create_publisher(
            JointTrajectory, '/position_commands', 10)

        # Control loop
        dt = 1.0 / self.get_parameter('control_rate').value
        self.control_timer = self.create_timer(dt, self._control_loop)

        # Command interface
        self.create_service(Trigger, '~/start_walking', self._start_walking_cb)

        self.get_logger().info('WBC Walking Node initialized')

    # ---------- Callbacks ----------

    def _joint_state_cb(self, msg: JointState):
        """Receive joint state from hardware/simulation."""
        with self.state_lock:
            for i, name in enumerate(self.JOINT_NAMES):
                if name in msg.name:
                    idx = msg.name.index(name)
                    self.q[i] = msg.position[idx]
                    if len(msg.velocity) > idx:
                        self.q_dot[i] = msg.velocity[idx]

    def _start_walking_cb(self, request, response):
        """Trigger walking pattern generation and execution."""
        n_steps = self.get_parameter('n_steps').value

        # Define initial foot positions (world frame)
        left_start = np.array([0.0, 0.06, 0.0, 0.0, 0.0, 0.0])
        right_start = np.array([0.0, -0.06, 0.0, 0.0, 0.0, 0.0])
        com_start = np.array([0.0, 0.0, self.get_parameter('com_height').value])

        self.get_logger().info(f'Generating {n_steps}-step walking pattern...')
        self.pattern = self.wpg.generate(n_steps, left_start, right_start, com_start)
        self.pattern_idx = 0
        self.is_walking = True
        self.walk_start_time = self.get_clock().now().nanoseconds / 1e9

        self.get_logger().info(
            f'Pattern ready: {len(self.pattern["timestamps"])} timesteps, '
            f'duration {self.pattern["timestamps"][-1]:.1f}s'
        )

        response.success = True
        response.message = f'Starting {n_steps}-step walk'
        return response

    # ---------- Control Loop ----------

    def _control_loop(self):
        """Main WBC control loop — runs at control_rate Hz."""
        if not self.is_walking or self.pattern is None:
            return

        now = self.get_clock().now().nanoseconds / 1e9
        elapsed = now - self.walk_start_time

        # Find pattern index
        while (self.pattern_idx < len(self.pattern["timestamps"]) - 1 and
               self.pattern["timestamps"][self.pattern_idx] < elapsed):
            self.pattern_idx += 1

        if self.pattern_idx >= len(self.pattern["timestamps"]) - 1:
            self.is_walking = False
            self.get_logger().info('Walking pattern complete — standing')
            return

        # Get desired states
        com_des = self.pattern["com_position"][self.pattern_idx]
        com_vel_des = self.pattern["com_velocity"][self.pattern_idx]
        lf_des = self.pattern["left_foot"][self.pattern_idx]
        rf_des = self.pattern["right_foot"][self.pattern_idx]
        lf_vel_des = np.zeros(6)
        rf_vel_des = np.zeros(6)

        # Determine support foot from foot heights
        lf_height = lf_des[2]
        rf_height = rf_des[2]
        eps = 0.005
        if lf_height < eps and rf_height < eps:
            support = "double"
        elif lf_height < eps:
            support = "left"
        elif rf_height < eps:
            support = "right"
        else:
            support = "double"  # both swinging (shouldn't happen)

        # Compute dynamics terms (use Pinocchio or similar in practice)
        with self.state_lock:
            q = self.q.copy()
            q_dot = self.q_dot.copy()

        M, h = self._compute_dynamics(q, q_dot)
        self.wbc.update_robot_state(q, q_dot, M, h)

        # Solve WBC
        try:
            q_ddot, tau = self.wbc.compute(
                com_des, com_vel_des,
                lf_des, rf_des,
                lf_vel_des, rf_vel_des,
                support_foot=support
            )
        except Exception as e:
            self.get_logger().warn(f'WBC solve failed: {e}')
            return

        # Publish commands
        self._publish_torque_commands(tau)
        # Alternative: integrate to position and publish

    # ---------- Dynamics (use Pinocchio or RBDL in practice) ----------

    def _compute_dynamics(self, q: np.ndarray, q_dot: np.ndarray):
        """
        Compute mass matrix M and bias forces h.

        PLACEHOLDER — replace with Pinocchio:
            import pinocchio
            pinocchio.crba(model, data, q)
            M = data.M
            pinocchio.nonLinearEffects(model, data, q, q_dot)
            h = data.nle

        Or use your URDF with:
            pinocchio.buildModelFromUrdf(urdf_path)
        """
        # Simplified model for illustration
        n = self.N_JOINTS
        M = np.eye(n)
        # Approximate leg masses from URDF
        M[0, 0] = 0.0105   # hip
        M[1, 1] = 0.004    # knee
        M[2, 2] = 0.0105
        M[3, 3] = 0.004
        M[4, 4] = 0.0015   # shoulder
        M[5, 5] = 0.0024
        M[6, 6] = 0.0015
        M[7, 7] = 0.0024

        # Gravity compensation (simplified)
        h = np.zeros(n)
        gravity_torque_hip = 5.0 * np.sin(q[0])  # right hip
        gravity_torque_knee = 2.0 * np.sin(q[1])
        h[0] = gravity_torque_hip
        h[1] = gravity_torque_knee
        h[2] = 5.0 * np.sin(q[2])  # left hip
        h[3] = 2.0 * np.sin(q[3])
        # Arms are light, approximate
        h[4] = 0.5 * np.sin(q[4])
        h[5] = 0.3 * np.sin(q[5])
        h[6] = 0.5 * np.sin(q[6])
        h[7] = 0.3 * np.sin(q[7])

        return M, h

    # ---------- Publishers ----------

    def _publish_torque_commands(self, tau: np.ndarray):
        """Publish joint efforts (torque control mode)."""
        msg = Float64MultiArray()
        msg.data = tau.tolist()
        self.torque_pub.publish(msg)

    def _publish_position_commands(self, q_des: np.ndarray):
        """Publish joint positions (position control mode, alternative)."""
        msg = JointTrajectory()
        msg.joint_names = self.JOINT_NAMES
        point = JointTrajectoryPoint()
        point.positions = q_des.tolist()
        point.time_from_start.sec = 0
        point.time_from_start.nanosec = int(0.02 * 1e9)
        msg.points = [point]
        self.position_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = WBCWalkingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()


import pinocchio
model = pinocchio.buildModelFromUrdf("human_urdf.urdf")
data = model.createData()
# Jacobienne du CoM
J_com = pinocchio.jacobianCenterOfMass(model, data, q)
# Jacobienne d'un pied (end-effector du knee_link)
J_foot = pinocchio.computeFrameJacobian(model, data, q, frame_id)
