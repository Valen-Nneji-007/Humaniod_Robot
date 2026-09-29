cmake_minimum_required(VERSION 3.8)
project(human_urdf)

find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)


install(
  PROGRAMS
    scripts/walk2.py
    scripts/walk.py
    scripts/gait_generator.py
    scripts/wbc_controller.py
    scripts/wbc_walking_node.py
    scripts/walking_pattern.py
  DESTINATION lib/${PROJECT_NAME}
)

foreach(dir config launch meshes urdf worlds)
  install(DIRECTORY ${dir}/
    DESTINATION share/${PROJECT_NAME}/${dir})
endforeach()

ament_package()