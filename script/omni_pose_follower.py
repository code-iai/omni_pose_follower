#!/usr/bin/env python
import traceback

import PyKDL
import rospy
from actionlib import SimpleActionServer
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal, JointTrajectoryControllerState
from geometry_msgs.msg import Twist, Quaternion, Pose
from sensor_msgs.msg import JointState
from tf.transformations import quaternion_about_axis
from urdf_parser_py.urdf import URDF


def pose_to_kdl(pose):
    """Convert a geometry_msgs Transform message to a PyKDL Frame.

    :param pose: The Transform message to convert.
    :type pose: Pose
    :return: The converted PyKDL frame.
    :rtype: PyKDL.Frame
    """
    return Frame(PyKDL.Rotation.Quaternion(pose.orientation.x,
                                           pose.orientation.y,
                                           pose.orientation.z,
                                           pose.orientation.w),
                 PyKDL.Vector(pose.position.x,
                              pose.position.y,
                              pose.position.z))


def kdl_to_twist(twist):
    """
    :type twist: PyKDL.Twist
    :return:
    """

    t = Twist()
    t.linear.x = twist.vel[0]
    t.linear.y = twist.vel[1]
    t.linear.z = twist.vel[2]
    t.angular.x = twist.rot[0]
    t.angular.y = twist.rot[1]
    t.angular.z = twist.rot[2]
    return t


def hacky_urdf_parser_fix(urdf_str):
    fixed_urdf = ''
    delete = False
    black_list = ['transmission', 'gazebo']
    black_open = ['<{}'.format(x) for x in black_list]
    black_close = ['</{}'.format(x) for x in black_list]
    for line in urdf_str.split('\n'):
        if len([x for x in black_open if x in line]) > 0:
            delete = True
        if len([x for x in black_close if x in line]) > 0:
            delete = False
            continue
        if not delete:
            fixed_urdf += line + '\n'
    return fixed_urdf


class Frame(PyKDL.Frame):
    def __sub__(self, other):
        return PyKDL.diff(other, self)


def plot_trajectory(goal, goal_vel, real, real_vel):
    """
    :type tj: Trajectory
    :param controlled_joints: only joints in this list will be added to the plot
    :type controlled_joints: list
    """
    import numpy as np
    import pylab as plt
    colors = [u'r', u'g', u'b']
    goal_positions = []
    real_positions = []
    goal_velocities = []
    real_velocities = []
    goal_times = []
    real_times = []
    names = ['odom_x', 'odom_y', 'odom_z']
    for time, point in goal:
        goal_positions.append([point.p[0], point.p[1], point.M.GetRotAngle()[0]])
        goal_times.append(time)
    for time, point in goal_vel:
        goal_velocities.append([point.vel[0], point.vel[1], point.rot[2]])
    goal_positions = np.array(goal_positions)
    goal_velocities = np.array(goal_velocities).T
    goal_times = np.array(goal_times)

    for time, point in real:
        real_positions.append([point.p[0], point.p[1], point.M.GetRotAngle()[0]])
        real_times.append(time)
    for time, point in real_vel:
        real_velocities.append([point.linear.x, point.linear.y, point.angular.z])
    real_positions = np.array(real_positions)
    real_velocities = np.array(real_velocities).T
    real_times = np.array(real_times)

    f, (ax1, ax2) = plt.subplots(2, sharex=True)
    ax1.set_title(u'position')
    ax2.set_title(u'velocity')
    for i, position in enumerate(goal_positions.T):
        ax1.plot(goal_times, position, colors[i] + '--', label=names[i])
        ax2.plot(goal_times, goal_velocities[i], colors[i] + '--', label=names[i])
    for i, position in enumerate(real_positions.T):
        ax1.plot(real_times, position, colors[i] + '', label=names[i])
        ax2.plot(real_times, real_velocities[i], colors[i] + '', label=names[i])
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width * 0.6, box.height])
    box = ax2.get_position()
    ax2.set_position([box.x0, box.y0, box.width * 0.6, box.height])

    # Put a legend to the right of the current axis
    ax1.legend(loc=u'center', bbox_to_anchor=(1.45, 0), prop={'size': 10})
    ax1.grid()
    ax2.grid()

    plt.show()


class OmniPoseFollower(object):

    def __init__(self):
        self.goal_traj = []
        self.real_traj = []

        urdf = rospy.get_param('robot_description')
        self.urdf = URDF.from_xml_string(hacky_urdf_parser_fix(urdf))

        limits = PyKDL.Twist()
        limits.vel[0] = self.urdf.joint_map[x_joint].limit.velocity
        limits.vel[1] = self.urdf.joint_map[y_joint].limit.velocity
        limits.rot[2] = self.urdf.joint_map[z_joint].limit.velocity
        self._min_output = -limits
        self._max_output = limits
        self.cmd = None

        self.cmd_vel_sub = rospy.Publisher('~cmd_vel', Twist, queue_size=10)

        js = rospy.wait_for_message('/joint_states', JointState)
        self.x_index_js = js.name.index(x_joint)
        self.y_index_js = js.name.index(y_joint)
        self.z_index_js = js.name.index(z_joint)
        self.current_goal = None
        self.js_sub = rospy.Subscriber('/joint_states', JointState, self.js_cb, queue_size=10)
        self.state_pub = rospy.Publisher('{}/state'.format(name_space), JointTrajectoryControllerState, queue_size=10)
        self.server = SimpleActionServer('{}/follow_joint_trajectory'.format(name_space),
                                         FollowJointTrajectoryAction,
                                         self.execute_cb,
                                         auto_start=False)
        self.server.start()

    def js_cb(self, js):
        self.current_pose = self.js_to_kdl(js.position[self.x_index_js],
                                           js.position[self.y_index_js],
                                           js.position[self.z_index_js])

        if self.cmd:
            if sim:
                cmd_msg = kdl_to_twist(self.cmd)
            else:
                cmd = self.current_pose.Inverse()*self.cmd
                cmd_msg = kdl_to_twist(cmd)
            self.real_traj.append([js.header.stamp.to_sec(), self.current_pose])
            self.real_traj_vel.append([js.header.stamp.to_sec(), self.cmd])
            self.cmd_vel_sub.publish(cmd_msg)

        state = JointTrajectoryControllerState()
        state.joint_names = [x_joint, y_joint, z_joint]
        self.state_pub.publish(state)

    def limit(self, x, ref):
        if x>0:
            return max(min(x, ref + vel_tolerance), ref - vel_tolerance)
        return min(max(x, ref - vel_tolerance), ref + vel_tolerance)

    def execute_cb(self, data):
        """
        :type data: FollowJointTrajectoryGoal
        :return:
        """
        try:
            self.start_time = rospy.get_rostime().to_sec()
            self.last_error = PyKDL.Twist()
            x_index = data.trajectory.joint_names.index(x_joint)
            y_index = data.trajectory.joint_names.index(y_joint)
            z_index = data.trajectory.joint_names.index(z_joint)
            i = 0
            self.hz = data.trajectory.points[1].time_from_start.to_sec() - \
                      data.trajectory.points[0].time_from_start.to_sec()
            time_tolerance = 0.1
            last_stamp = 0
            self.goal_traj = []
            self.real_traj = []
            self.goal_traj_vel = []
            self.real_traj_vel = []
            while i < len(data.trajectory.points) and not self.server.is_preempt_requested():
                current_point = data.trajectory.points[i]
                time_from_start = rospy.get_rostime().to_sec() - self.start_time
                time_from_start2 = current_point.time_from_start.to_sec()
                if time_from_start < time_from_start2:
                    self.current_goal = self.js_to_kdl(current_point.positions[x_index],
                                                       current_point.positions[y_index],
                                                       current_point.positions[z_index])
                    self.current_goal_vel = PyKDL.Twist()
                    self.current_goal_vel.vel[0] = current_point.velocities[x_index]
                    self.current_goal_vel.vel[1] = current_point.velocities[y_index]
                    self.current_goal_vel.rot[2] = current_point.velocities[z_index]

                    dt = time_from_start2 - last_stamp
                    last_stamp = time_from_start2

                    error = (self.current_goal - self.current_pose)

                    cmd = error / dt
                    cmd.vel[0] = self.limit(cmd.vel[0], self.current_goal_vel.vel[0])
                    cmd.vel[1] = self.limit(cmd.vel[1], self.current_goal_vel.vel[1])
                    cmd.rot[2] = self.limit(cmd.rot[2], self.current_goal_vel.rot[2])
                    self.cmd = cmd
                    self.goal_traj.append([self.start_time + time_from_start2, self.current_goal])
                    self.goal_traj_vel.append([self.start_time + time_from_start2, self.current_goal_vel])
                else:
                    i += 1
                tfs = rospy.get_rostime().to_sec() - self.start_time
                asdf = current_point.time_from_start.to_sec() - tfs
                rospy.sleep(asdf)
            # rospy.sleep(time_tolerance)
            self.cmd = None
            rospy.loginfo('goal reached, final diff: {}'.format(PyKDL.diff(self.current_pose, self.current_goal)))
            # plot_trajectory(self.goal_traj, self.goal_traj_vel, self.real_traj, self.real_traj_vel)
            self.current_goal = None
            self.server.set_succeeded()
        except:
            traceback.print_exc()
            rospy.loginfo('aborted current goal')
            self.server.set_aborted()
        finally:
            self.current_goal = None
            self.cmd_vel_sub.publish(Twist())

    def js_to_kdl(self, x, y, rot):
        ps = Pose()
        ps.position.x = x
        ps.position.y = y
        ps.orientation = Quaternion(*quaternion_about_axis(rot, [0, 0, 1]))
        return pose_to_kdl(ps)


if __name__ == '__main__':
    try:
        rospy.init_node('omni_pose_follower')
        name_space = rospy.get_param('~name_space')
        sim = rospy.get_param('~sim')
        vel_tolerance = rospy.get_param('~vel_tolerance')
        x_joint = rospy.get_param('{}/odom_x_joint'.format(name_space))
        y_joint = rospy.get_param('{}/odom_y_joint'.format(name_space))
        z_joint = rospy.get_param('{}/odom_z_joint'.format(name_space))
        opf = OmniPoseFollower()
        rospy.loginfo('pose follower running')
        rospy.spin()

    except rospy.ROSInterruptException:
        pass
