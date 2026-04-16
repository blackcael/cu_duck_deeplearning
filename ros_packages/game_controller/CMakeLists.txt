cmake_minimum_required(VERSION 3.0.2)
project(game_controller)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/controller.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
