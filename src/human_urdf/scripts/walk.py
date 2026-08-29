#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import math


class WalkingController(Node):

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

        self.timer = self.create_timer(0.1, self.walk)

        self.t = 0.0

    def walk(self):

        

        step_period = 4.0

        hip_amplitude = 0.20
        knee_amplitude = 0.35

        phase = 2.0 * math.pi * (self.t / step_period)

        # Opposite leg phases
        waistR = hip_amplitude * math.sin(phase)
        waistL = -hip_amplitude * math.sin(phase)

        # Knees bend when the corresponding leg swings forward
        kneeR = max(0.0, knee_amplitude * math.sin(phase))
        kneeL = max(0.0, -knee_amplitude * math.sin(phase))

        msg = JointTrajectory()

        msg.joint_names = self.joints

        point = JointTrajectoryPoint()

        point.positions = [
            0.0,       # shoulderR
            0.0,       # elbowR
            0.0,       # shoulderL
            0.0,       # elbowL
            0.0,       # neck
            waistR,    # waistR
            kneeR,     # kneeR
            waistL,    # waistL
            kneeL      # kneeL
        ]

        point.time_from_start.sec = 0
        point.time_from_start.nanosec = 100000000

        msg.points.append(point)

        self.publisher.publish(msg)

        self.t += 0.1


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