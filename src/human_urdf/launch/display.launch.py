import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    pkg_share = get_package_share_directory('human_urdf')

    # Xacro file
    urdf_file = os.path.join(
        pkg_share,
        'urdf',
        'human_urdf.urdf.xacro'
    )

    # RViz configuration
    rviz_config_file = os.path.join(
        pkg_share,
        'urdf',
        'urdf.rviz'
    )

    # Process Xacro -> URDF
    robot_description = Command([
        'xacro ',
        urdf_file
    ])

    return LaunchDescription([

        DeclareLaunchArgument(
            'model',
            default_value=urdf_file,
            description='Path to robot Xacro file'
        ),

        # Joint State Publisher GUI
        Node(
            package='joint_state_publisher_gui',
            executable='joint_state_publisher_gui',
            name='joint_state_publisher_gui',
            output='screen'
        ),

        # Robot State Publisher
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[
                {
                    'robot_description': robot_description
                }
            ]
        ),

        # RViz
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=[
                '-d',
                rviz_config_file
            ]
        ),
    ])