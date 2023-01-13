cmake_minimum_required(VERSION 3.0.2)
project(leo_docking)

find_package(catkin REQUIRED COMPONENTS
  actionlib_msgs
  message_generation
)

add_action_files(DIRECTORY action FILES PerformDocking.action)

catkin_python_setup()

generate_messages(DEPENDENCIES actionlib_msgs)

catkin_package(
  CATKIN_DEPENDS
    actionlib_msgs
    message_runtime
)

catkin_install_python(
  PROGRAMS
  scripts/docking_main
  scripts/docking_client
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)