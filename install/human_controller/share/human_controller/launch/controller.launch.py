
from launch import LaunchDescription
from launch_ros.actions import Node
import os

from launch_ros.parameter_descriptions import ParameterValue
from launch.substitutions import Command, LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument
from launch.conditions import UnlessCondition


def generate_launch_description():

    # Simulation argument
    is_sim_arg = DeclareLaunchArgument(
        "is_sim",
        default_value="True"
    )

    is_sim = LaunchConfiguration("is_sim")

    # Robot description
    robot_description = ParameterValue(
        Command([
            "xacro ",
            os.path.join(
                get_package_share_directory("human_urdf"),
                "urdf",
                "human_urdf.urdf"
            ),
            " is_sim:=",
            is_sim
        ]),
        value_type=str,
    )

    # Robot State Publisher
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[
            {
                "robot_description": robot_description,
                "use_sim_time": is_sim
            }
        ]
    )

    # ros2_control controller manager
    control_manager = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[
            {
                "robot_description": robot_description,
                "use_sim_time": is_sim
            },
            os.path.join(
                get_package_share_directory("human_controller"),
                "config",
                "human_controllers.yaml"
            )
        ],
        condition=UnlessCondition(is_sim)
    )

    # Joint State Broadcaster
    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_state_broadcaster",
            "--controller-manager",
            "/controller_manager"
        ]
    )

    joint_trajectory_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_trajectory_controller",
            "--controller-manager",
            "/controller_manager"
    ]
)

    return LaunchDescription([
        is_sim_arg,
        robot_state_publisher,
        control_manager,
        joint_state_broadcaster_spawner,
        joint_trajectory_controller_spawner
    ])

