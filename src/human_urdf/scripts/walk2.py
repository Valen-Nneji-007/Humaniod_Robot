#!/usr/bin/env python3

import math

import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


class WalkingController(Node):
    """
    Simple sinusoidal CPG (central pattern generator) gait.

    Legs move in opposite phase; knees fold only during that leg's forward
    swing. Arms swing opposite the same-side leg (contralateral coordination),
    which is the main visual cue that reads as "human" rather than "robot".
    A short startup ramp eases the amplitude in from zero instead of snapping
    straight to full stride.
    """

    def __init__(self):
        super().__init__('walking_controller')

        self.publisher = self.create_publisher(
            JointTrajectory,
            '/body_controller/joint_trajectory',
            10
        )

        self.joints = [
            'joint_shoulderR',
            'joint_elbowR',
            'joint_shoulderL',
            'joint_elbowL',
            'joint_neck',
            'joint_waistR',
            'joint_kneeR',
            'joint_waistL',
            'joint_kneeL'
        ]

        # Gait parameters - tune live without touching code, e.g.:
        #   ros2 param set /walking_controller step_period 1.2
        self.declare_parameter('step_period', 4.0)       # seconds per full stride
        self.declare_parameter('hip_amplitude', 0.20)    # rad
        self.declare_parameter('knee_amplitude', 0.35)   # rad
        self.declare_parameter('arm_amplitude', 0.30)    # rad
        self.declare_parameter('elbow_amplitude', 0.15)  # rad
        self.declare_parameter('ramp_up_time', 2.0)      # seconds to reach full stride

        self.dt = 0.1
        self.timer = self.create_timer(self.dt, self.walk)

        self.t = 0.0        # phase clock, wrapped to [0, step_period)
        self.elapsed = 0.0  # unwrapped clock, used only for the startup ramp

    @staticmethod
    def _smoothstep(x):
        x = max(0.0, min(1.0, x))
        return x * x * (3.0 - 2.0 * x)

    def walk(self):

        step_period = self.get_parameter('step_period').value
        hip_amplitude = self.get_parameter('hip_amplitude').value
        knee_amplitude = self.get_parameter('knee_amplitude').value
        arm_amplitude = self.get_parameter('arm_amplitude').value
        elbow_amplitude = self.get_parameter('elbow_amplitude').value
        ramp_up_time = self.get_parameter('ramp_up_time').value

        phase = 2.0 * math.pi * (self.t / step_period)
        ramp = self._smoothstep(self.elapsed / ramp_up_time) if ramp_up_time > 0.0 else 1.0
        sin_p = math.sin(phase)

        # Legs: opposite phase. Squaring the clamped sine keeps the same
        # peak knee bend as before but removes the velocity kink that a
        # bare max(0, sin(...)) has at each swing transition.
        waistR = ramp * hip_amplitude * sin_p
        waistL = -waistR
        kneeR = ramp * knee_amplitude * max(0.0, sin_p) ** 2
        kneeL = ramp * knee_amplitude * max(0.0, -sin_p) ** 2

        # Arms: swing opposite the same-side leg, with a little elbow
        # bend on the backswing.
        shoulderR = -ramp * arm_amplitude * sin_p
        shoulderL = ramp * arm_amplitude * sin_p
        elbowR = ramp * elbow_amplitude * max(0.0, sin_p) ** 2
        elbowL = ramp * elbow_amplitude * max(0.0, -sin_p) ** 2

        msg = JointTrajectory()
        msg.joint_names = self.joints

        point = JointTrajectoryPoint()
        point.positions = [
            shoulderR,
            elbowR,
            shoulderL,
            elbowL,
            0.0,     # neck: keep the head level
            waistR,
            kneeR,
            waistL,
            kneeL
        ]

        point.time_from_start.sec = 0
        point.time_from_start.nanosec = int(self.dt * 1e9)

        msg.points.append(point)
        self.publisher.publish(msg)

        self.t = (self.t + self.dt) % step_period
        self.elapsed += self.dt


def main(args=None):

    rclpy.init(args=args)

    node = WalkingController()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()