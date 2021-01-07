cmake_minimum_required(VERSION 3.0.2)
project(omni_pose_follower)

find_package(catkin REQUIRED COMPONENTS)

catkin_package(DEPENDS)

include_directories(
# include
# ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS 
	script/omni_pose_follower.py
	script/omni_pose_follower_map.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})