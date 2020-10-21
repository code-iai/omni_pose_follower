#! /usr/bin/env python
import rospy

import actionlib

from control_msgs.msg import JointTrajectoryControllerState
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryActionFeedback, FollowJointTrajectoryActionResult, FollowJointTrajectoryActionGoal
from geometry_msgs.msg import Twist, Pose, Quaternion, TransformStamped
from sensor_msgs.msg import JointState
import tf
from tf2_ros import TransformException
import PyKDL

import math

class BaseControl(object):
  # create messages that are used to publish state/feedback/result
  _state = JointTrajectoryControllerState()
  _feedback = FollowJointTrajectoryActionFeedback()
  _result = FollowJointTrajectoryActionResult()

  def __init__(self):
    self.cmd_vel_pub = rospy.Publisher('~cmd_vel', Twist, queue_size=10)

    joint_states = rospy.wait_for_message('base/joint_states', JointState)
    try:
      self.odom_x_joint_index = joint_states.name.index(odom_x_joint)
      self.odom_y_joint_index = joint_states.name.index(odom_y_joint)
      self.odom_z_joint_index = joint_states.name.index(odom_z_joint)

      rospy.loginfo("omni_pose_follower found odom joints")
    except ValueError as e:
      rospy.logwarn("omni_pose_follower couldn't find odom joints in joint states!")
      return

    # create tf listener
    self.tf_listener = tf.TransformListener()

    # set frames
    self.set_odom_origin()

    # create state publisher
    self.state_pub = rospy.Publisher('{}/state'.format(name_space), JointTrajectoryControllerState, queue_size=10)
    self.state_sub = rospy.Subscriber('base/joint_states', JointState, self.joint_states_callback, queue_size=10)

    # create the action server
    self._as = actionlib.SimpleActionServer('{}/follow_joint_trajectory'.format(name_space), FollowJointTrajectoryAction, self.goal_callback, False)
    self._as.start()

  def set_odom_origin(self):
    try:
      self.tf_listener.waitForTransform("map", odom, rospy.Time(), rospy.Duration(1))
    except TransformException as e:
      rospy.logwarn("omni_pose_follower couldn't find odom frame")
    else:
      t = self.tf_listener.getLatestCommonTime("map", odom)
      pos, quat = self.tf_listener.lookupTransform("map", odom, t)
      self.map_T_odom_origin = PyKDL.Frame(PyKDL.Rotation.Quaternion(quat[0], quat[1], quat[2], quat[3]), PyKDL.Vector(pos[0], pos[1], pos[2]))

  def joint_states_callback(self, joint_states):
    try:
      self.tf_listener.waitForTransform("map", base, rospy.Time(), rospy.Duration(1))
    except TransformException as e:
      rospy.logwarn("omni_pose_follower couldn't find map frame or base_footprint frame")
    else:
      t = self.tf_listener.getLatestCommonTime("map", base)
      pos, quat = self.tf_listener.lookupTransform("map", base, t)
      map_T_base_footprint = PyKDL.Frame(PyKDL.Rotation.Quaternion(quat[0], quat[1], quat[2], quat[3]), PyKDL.Vector(pos[0], pos[1], pos[2]))

      odom_origin_T_map = self.map_T_odom_origin.Inverse()
      self.odom_origin_T_base_footprint = odom_origin_T_map * map_T_base_footprint

      pos_x = self.odom_origin_T_base_footprint.p.x()
      pos_y = self.odom_origin_T_base_footprint.p.y()
      [_, _, euler_z] = self.odom_origin_T_base_footprint.M.GetRPY()
      self._state.actual.positions = [pos_x, pos_y, euler_z]

    try:
      self._state.actual.velocities = [joint_states.velocity[self.odom_x_joint_index], joint_states.velocity[self.odom_y_joint_index], joint_states.velocity[self.odom_z_joint_index]]
    except IndexError as e:
      rospy.logwarn("omni_pose_follower couldn't find enough odom joint velocities in joint states!")
      self._as.set_aborted()
      return
    
    # publish joint_states
    self._state.header.stamp = rospy.Time.now()
    self._state.header.frame_id = joint_states.header.frame_id
    self._state.joint_names = joint_states.name
    self.state_pub.publish(self._state)

  def goal_callback(self, goal):
    # helper variables
    success = True
    rate = rospy.Rate(freq)
    
    try:
      goal_odom_x_joint_index = goal.trajectory.joint_names.index(odom_x_joint)
      goal_odom_y_joint_index = goal.trajectory.joint_names.index(odom_y_joint)
      goal_odom_z_joint_index = goal.trajectory.joint_names.index(odom_z_joint)
    except:
      rospy.loginfo("omni_pose_follower aborted current goal")
      self._as.set_aborted()
      return
    
    # check if the path is not too short
    L = len(goal.trajectory.points)
    s_x = sum([abs(goal.trajectory.points[t].velocities[goal_odom_x_joint_index]) for t in range(L)])
    s_y = sum([abs(goal.trajectory.points[t].velocities[goal_odom_y_joint_index]) for t in range(L)])
    s_z = sum([abs(goal.trajectory.points[t].velocities[goal_odom_z_joint_index]) for t in range(L)])
    if s_x < S_x and s_y < S_y and s_z < S_z: # path is too short
      rospy.loginfo("The goal has been reached")
      self._result.result.error_string = "no error"
      self._as.set_succeeded(self._result.result)
      return

    # initialization
    cmd_vel = Twist()
    t = 0
    delta_T = 1/freq
    time_start = goal.trajectory.header.stamp - rospy.Duration(T_delay)
    self.set_odom_origin()

    while True:
      if self._as.is_preempt_requested():
        rospy.loginfo("The goal has been preempted")
        # the following line sets the client in preempted state (goal cancelled)
        self._as.set_preempted()
        success = False
        break

      time_from_start = rospy.Time.now() - time_start 

      # set goal velocites in map frame
      if t < L and t >= 0:
        for point_index in range(t,L):
          if goal.trajectory.points[point_index].time_from_start < time_from_start:
            t += 1
          else:
            break

      if t == L:
        t = -1
        time_finish = rospy.Time.now()

      if t == -1:
        time_from_finish = rospy.Time.now() - time_finish
        if time_from_finish.secs > T_finish:
          success = True
          break

      v_x = goal.trajectory.points[t].velocities[goal_odom_x_joint_index]
      v_y = goal.trajectory.points[t].velocities[goal_odom_y_joint_index]
      v_z = goal.trajectory.points[t].velocities[goal_odom_z_joint_index]

      pos = [goal.trajectory.points[t].positions[goal_odom_x_joint_index], goal.trajectory.points[t].positions[goal_odom_y_joint_index], 0]
      quat = tf.transformations.quaternion_from_euler(0, 0, goal.trajectory.points[t].positions[goal_odom_z_joint_index])
      odom_origin_T_base_footprint_goal = PyKDL.Frame(PyKDL.Rotation.Quaternion(quat[0], quat[1], quat[2], quat[3]), PyKDL.Vector(pos[0], pos[1], pos[2]))

      base_footprint_T_odom_origin = self.odom_origin_T_base_footprint.Inverse()
      base_footprint_T_base_footprint_goal = base_footprint_T_odom_origin * odom_origin_T_base_footprint_goal
      
      error_odom_x_pos = base_footprint_T_base_footprint_goal.p.x()
      error_odom_y_pos = base_footprint_T_base_footprint_goal.p.y()
      [_,_,error_odom_z_pos] = base_footprint_T_base_footprint_goal.M.GetRPY()

      # goal error
      if t == -1:
        if abs(error_odom_x_pos) + abs(error_odom_y_pos) + abs(error_odom_z_pos) < 0.001:
          success = True
          break
      # print("At " + str(t) + ": error_traj = " + str([error_odom_x_pos, error_odom_y_pos, error_odom_z_pos]))

      error_odom_x_vel = goal.trajectory.points[t].velocities[goal_odom_x_joint_index] - self._state.actual.velocities[0]
      error_odom_y_vel = goal.trajectory.points[t].velocities[goal_odom_y_joint_index] - self._state.actual.velocities[1]
      error_odom_z_vel = goal.trajectory.points[t].velocities[goal_odom_z_joint_index] - self._state.actual.velocities[2]

      self._feedback.feedback.header.stamp = rospy.Time.now()
      self._feedback.feedback.header.frame_id = self._state.header.frame_id
      self._feedback.feedback.joint_names = self._state.joint_names
      self._feedback.feedback.desired.positions = goal.trajectory.points[t].positions
      self._feedback.feedback.desired.velocities = goal.trajectory.points[t].velocities
      self._feedback.feedback.desired.time_from_start = time_from_start
      self._feedback.feedback.actual.positions = self._state.actual.positions
      self._feedback.feedback.actual.velocities = self._state.actual.velocities
      self._feedback.feedback.actual.time_from_start = time_from_start
      self._feedback.feedback.error.positions = [error_odom_x_pos, error_odom_y_pos, error_odom_z_pos]
      self._feedback.feedback.error.velocities = [error_odom_x_vel, error_odom_y_vel, error_odom_z_vel]
      self._feedback.feedback.error.time_from_start = time_from_start
      
      # publish the feedback
      self._as.publish_feedback(self._feedback.feedback)

      # transform velocities from map frame to base frame and add feedback control
      sin_z = math.sin(self._state.actual.positions[2])
      cos_z = math.cos(self._state.actual.positions[2])
      
      cmd_vel_x = v_x * cos_z + v_y * sin_z + K_x['p'] * error_odom_x_pos + K_x['d'] * error_odom_x_vel
      cmd_vel_y = -v_x * sin_z + v_y * cos_z + K_y['p'] * error_odom_y_pos + K_y['d'] * error_odom_y_vel
      cmd_vel_z = v_z + K_z['p'] * error_odom_z_pos + K_z['d'] * error_odom_z_vel

      # add time delay
      cmd_vel.linear.x = T_delay/(T_delay+delta_T) * cmd_vel.linear.x + delta_T/(T_delay+delta_T) * cmd_vel_x
      cmd_vel.linear.y = T_delay/(T_delay+delta_T) * cmd_vel.linear.y + delta_T/(T_delay+delta_T) * cmd_vel_y
      cmd_vel.angular.z = T_delay/(T_delay+delta_T) * cmd_vel.angular.z + delta_T/(T_delay+delta_T) * cmd_vel_z

      # publish the velocity
      self.cmd_vel_pub.publish(cmd_vel)
      rate.sleep()

    # set velocites to zero
    self.cmd_vel_pub.publish(Twist())

    if success:
      rospy.loginfo("The goal has been reached, final diff: {}".format([error_odom_x_pos, error_odom_y_pos, error_odom_z_pos]))
      self._result.result.error_string = "no error"
      self._as.set_succeeded(self._result.result)
    else:
      self._as.set_aborted(self._result.result)

if __name__ == '__main__':
  rospy.init_node("omni_pose_follower")
  name_space = rospy.get_param('~name_space')
  odom_x_joint = rospy.get_param('{}/odom_x_joint'.format(name_space))
  odom_y_joint = rospy.get_param('{}/odom_y_joint'.format(name_space))
  odom_z_joint = rospy.get_param('{}/odom_z_joint'.format(name_space))

  odom = rospy.get_param('{}/odom_frame'.format(name_space))
  base = rospy.get_param('{}/base_frame'.format(name_space))
  K_x = rospy.get_param('{}/K_x'.format(name_space))
  K_y = rospy.get_param('{}/K_y'.format(name_space))
  K_z = rospy.get_param('{}/K_z'.format(name_space))
  freq = rospy.get_param('{}/freq'.format(name_space))
  T_delay = rospy.get_param('{}/T_delay'.format(name_space))
  T_finish = rospy.get_param('{}/T_finish'.format(name_space))
  S_x = rospy.get_param('{}/S_x'.format(name_space))
  S_y = rospy.get_param('{}/S_y'.format(name_space))
  S_z = rospy.get_param('{}/S_z'.format(name_space))

  # publish info to the console for the user
  rospy.loginfo("omni_pose_follower starts")

  # start the base control
  BaseControl()

  # keep it running
  rospy.spin()
