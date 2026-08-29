import os
from os import pathsep
from pathlib import Path

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable, RegisterEventHandler
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource

from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch.event_handlers import OnProcessExit


def generate_launch_description():

    human_urdf = get_package_share_directory("human_urdf")

    # ---------------------------------------------------------
    # Robot model
    # ---------------------------------------------------------

    model_arg = DeclareLaunchArgument(
        name="model",
        default_value=os.path.join(
            human_urdf,
            "urdf",
            "human_urdf.urdf.xacro"
        ),
        description="Path to the robot URDF/Xacro file"
    )

    # ---------------------------------------------------------
    # Gazebo world
    # ---------------------------------------------------------

    world_name_arg = DeclareLaunchArgument(
        name="world_name",
        default_value="empty.sdf",
        description="Gazebo world file"
    )

    world_path = PathJoinSubstitution([
        human_urdf,
        "worlds",
        LaunchConfiguration("world_name")
    ])

    # ---------------------------------------------------------
    # Gazebo resource path
    # ---------------------------------------------------------

    model_path = str(Path(human_urdf).parent.resolve())

    model_path += pathsep + os.path.join(
        get_package_share_directory("human_urdf"),
        "model"
    )

    gazebo_resource_path = SetEnvironmentVariable(
        "GZ_SIM_RESOURCE_PATH",
        model_path
    )

    # ---------------------------------------------------------
    # ROS distribution
    # ---------------------------------------------------------

    ros_distro = os.environ["ROS_DISTRO"]

    is_ignition = "True" if ros_distro == "humble" else "False"

    # ---------------------------------------------------------
    # Robot description
    # ---------------------------------------------------------

    robot_description = ParameterValue(
        Command([
            "xacro ",
            LaunchConfiguration("model"),
            " is_ignition:=",
            is_ignition
        ]),
        value_type=str
    )

    # ---------------------------------------------------------
    # Robot State Publisher
    # ---------------------------------------------------------

    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[
            {
                "robot_description": robot_description,
                "use_sim_time": True
            }
        ]
    )

    # ---------------------------------------------------------
    # Gazebo
    # ---------------------------------------------------------

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("ros_gz_sim"),
                "launch",
                "gz_sim.launch.py"
            )
        ),
        launch_arguments={
            "gz_args": [world_path, " -v 4 -r"]
        }.items()
    )

    # ---------------------------------------------------------
    # Spawn robot
    # ---------------------------------------------------------

    gz_spawn_entity = Node(
        package="ros_gz_sim",
        executable="create",
        output="screen",
        arguments=[
            "-topic",
            "robot_description",
            "-name",
            "human_urdf",
            "-z",
            "0.05"
        ]
    )

    # ---------------------------------------------------------
    # Gazebo <-> ROS 2 bridge
    # ---------------------------------------------------------

    gz_ros2_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
            "/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan",
            '/left_camera@sensor_msgs/msg/Image@gz.msgs.Image', 
            '/right_camera@sensor_msgs/msg/Image@gz.msgs.Image'
        ],
        output="screen"
    )

    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster"],
    )


    body_controller_spawner = Node(
    package="controller_manager",
    executable="spawner",
    arguments=["body_controller"],
    )

    delayed_jsb = RegisterEventHandler(
        event_handler=OnProcessExit(
        target_action=gz_spawn_entity,
        on_exit=[joint_state_broadcaster_spawner],
    )
    )

    delayed_controller = RegisterEventHandler(
        event_handler=OnProcessExit(
        target_action=joint_state_broadcaster_spawner,
        on_exit=[body_controller_spawner],
    ))

    # ---------------------------------------------------------
    # Launch everything
    # ---------------------------------------------------------

    return LaunchDescription([
        model_arg,
        world_name_arg,
        gazebo_resource_path,
        robot_state_publisher_node,
        delayed_controller,
        delayed_jsb,
        gazebo,
        gz_spawn_entity,
        gz_ros2_bridge
    ])