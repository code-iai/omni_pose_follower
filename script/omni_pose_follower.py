#!/usr/bin/env python
import numpy as np
import traceback

import PyKDL
import rospy
from actionlib import SimpleActionServer
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal, JointTrajectoryControllerState
from geometry_msgs.msg import Twist, Pose, Quaternion
from scipy.interpolate import interp1d
from sensor_msgs.msg import JointState
from tf.transformations import unit_vector, quaternion_about_axis
from urdf_parser_py.urdf import URDF


# TODO possible improvements
# spline interpolation on original traj
# limit

def pose_to_kdl(pose):
    """Convert a geometry_msgs Transform message to a PyKDL Frame.
    :param pose: The Transform message to convert.
    :type pose: Pose
    :return: The converted PyKDL frame.
    :rtype: PyKDL.Frame
    """
    return PyKDL.Frame(PyKDL.Rotation.Quaternion(pose.orientation.x,
                                                 pose.orientation.y,
                                                 pose.orientation.z,
                                                 pose.orientation.w),
                       PyKDL.Vector(pose.position.x,
                                    pose.position.y,
                                    pose.position.z))


def js_to_kdl(x, y, rot):
    ps = Pose()
    ps.position.x = x
    ps.position.y = y
    ps.orientation = Quaternion(*quaternion_about_axis(rot, [0, 0, 1]))
    return pose_to_kdl(ps)


def make_twist(x, y, rot):
    t = PyKDL.Twist()
    t.vel[0] = x
    t.vel[1] = y
    t.rot[2] = rot
    return t


def kdl_to_msg(twist):
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


def np_to_msg(array):
    t = Twist()
    t.linear.x = array[0]
    t.linear.y = array[1]
    t.angular.z = array[2]
    return t


def kdl_to_np(thing):
    if isinstance(thing, PyKDL.Twist):
        return np.array([thing.vel[0], thing.vel[1], thing.rot[2]])


def np_to_twist(array):
    t = PyKDL.Twist()
    t.vel[0] = array[0]
    t.vel[1] = array[1]
    t.rot[2] = array[2]
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


_EPS = np.finfo(float).eps * 4.0


def angular_distance(v1, v2):
    d = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    if abs(abs(d) - 1.0) < _EPS:
        return 0
    return np.arccos(d)


def make_cmd(current_pose, goal_pose, current_vel, goal_vels, current_time, eps=1e-5):
    goal_vel = goal_vels(current_time)
    pose_error = goal_pose - current_pose
    if np.linalg.norm(current_vel) == 0:
        return goal_vel
    else:
        goal_vel2 = goal_vels(current_time - 0.083)
        reference_angle = angular_distance(goal_vel2, goal_vel)
        needed_angle = angular_distance(current_vel, pose_error)
        beta = min(needed_angle, reference_angle + 0.01)
        cmd = interpolate(current_vel, pose_error, beta)
        if np.linalg.norm(pose_error) < eps:
            return np.array([0, 0, 0])
        cmd = cmd * max(np.linalg.norm(goal_vel), min(np.linalg.norm(pose_error), 0.1))
        return goal_vel


def interpolate(q0, q1, beta):
    if np.linalg.norm(q0) < _EPS:
        return q1
    q0 = unit_vector(q0)
    q1 = unit_vector(q1)
    if beta == 0.0:
        return q0
    d = np.dot(q0, q1)
    angle = np.arccos(d)
    if abs(angle) < _EPS:
        return q0
    fraction = beta / angle
    isin = 1.0 / np.sin(angle)
    q0 *= np.sin((1.0 - fraction) * angle) * isin
    q1 *= np.sin(fraction * angle) * isin
    q0 += q1
    return q0


def traj_to_poses(traj, x_index, y_index, z_index):
    position_traj = []
    vel_traj = []
    time_traj = []
    for current_point in traj.trajectory.points:
        position_traj.append(np.array([current_point.positions[x_index],
                                       current_point.positions[y_index],
                                       current_point.positions[z_index]]))
        vel_traj.append(np.array([current_point.velocities[x_index],
                                  current_point.velocities[y_index],
                                  current_point.velocities[z_index]]))
        time_traj.append(current_point.time_from_start.to_sec())
    dt = abs(time_traj[-1] - time_traj[-2])
    asdf = time_traj[-1]
    for i in range(20):
        asdf += dt
        position_traj.append(position_traj[-1])
        vel_traj.append(vel_traj[-1])
        time_traj.append(asdf)
    position_traj = np.array(position_traj)
    vel_traj = np.array(vel_traj)
    time_traj = np.array(time_traj)

    dtx = interp1d(time_traj, vel_traj[:, 0])
    dty = interp1d(time_traj, vel_traj[:, 1])
    dtz = interp1d(time_traj, vel_traj[:, 2])

    def magic(x):
        x = max(0, x)
        return np.array([dtx(x), dty(x), dtz(x)])

    vel_traj2 = magic

    fx = interp1d(time_traj, position_traj[:, 0])
    fy = interp1d(time_traj, position_traj[:, 1])
    fz = interp1d(time_traj, position_traj[:, 2])

    def magic2(x):
        x = max(0, x)
        return np.array([fx(x), fy(x), fz(x)])

    position_traj2 = magic2

    return position_traj2, vel_traj2, time_traj


class OmniPoseFollower(object):

    def __init__(self):
        self.goal_traj = []
        self.real_traj = []
        self.acc_limit = PyKDL.Twist()
        self.acc_limit.vel[0] = 0.01
        self.acc_limit.vel[1] = 0.01
        self.acc_limit.rot[2] = 0.01
        self.done = True

        urdf = rospy.get_param('robot_description')
        self.urdf = URDF.from_xml_string(hacky_urdf_parser_fix(urdf))

        limits = PyKDL.Twist()
        limits.vel[0] = self.urdf.joint_map[x_joint].limit.velocity
        limits.vel[1] = self.urdf.joint_map[y_joint].limit.velocity
        limits.rot[2] = self.urdf.joint_map[z_joint].limit.velocity
        self._min_vel = -limits
        self._max_vel = limits
        self.cmd = None
        self.last_cmd = PyKDL.Twist()
        self.pose_history = None

        self.cmd_vel_sub = rospy.Publisher('~cmd_vel', Twist, queue_size=10)
        self.debug_vel = rospy.Publisher('debug_vel', Twist, queue_size=10)

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
        self.current_pose = np.array([js.position[self.x_index_js],
                                      js.position[self.y_index_js],
                                      js.position[self.z_index_js]])
        try:
            current_vel = np.array([js.velocity[self.x_index_js],
                                    js.velocity[self.y_index_js],
                                    js.velocity[self.z_index_js]])
        except IndexError as e:
            rospy.logwarn('velocity entry in joint state is empty')
            return
        time = js.header.stamp.to_sec()
        self.last_cmd = current_vel
        if not self.done:
            time_from_start = time - self.start_time
            if time_from_start > 0:

                goal_pose = self.pose_traj2(time_from_start)
                cmd = make_cmd(self.current_pose, goal_pose, self.last_cmd, self.goal_vel2, time_from_start)
                if np.any(np.isnan(cmd)):
                    print('fuck')
                else:
                    cmd = self.limit_vel(cmd)
                    cmd = kdl_to_np(js_to_kdl(*self.current_pose).M.Inverse() * make_twist(*cmd))
                    self.debug_vel.publish(np_to_msg(cmd))
                    # cmd = self.hack(cmd)

                    cmd_msg = np_to_msg(cmd)
                    self.cmd_vel_sub.publish(cmd_msg)
                    self.last_cmd = cmd

        state = JointTrajectoryControllerState()
        state.joint_names = [x_joint, y_joint, z_joint]
        self.state_pub.publish(state)

    def hack(self, vel, eps=0.005):
        asdf = np.array([0.015, 0.015, 0.04])
        vel[np.abs(vel) < eps] = 0
        vel += np.sign(vel) * asdf
        return vel

    def limit_correction(self, x, ref):
        if x > 0:
            return max(min(x, ref + vel_tolerance), ref - vel_tolerance)
        return min(max(x, ref - vel_tolerance), ref + vel_tolerance)

    def limit_vel(self, vel):
        vel[0] = max(min(vel[0], self._max_vel.vel[0]), self._min_vel.vel[0])
        vel[1] = max(min(vel[1], self._max_vel.vel[1]), self._min_vel.vel[1])
        vel[2] = max(min(vel[2], self._max_vel.rot[2]), self._min_vel.rot[2])
        return vel

    def limit_acceleration(self, old_vel, new_vel):
        vel = np.array([old_vel[0] + max(-self.acc_limit.vel[0],
                                         min(self.acc_limit.vel[0], new_vel[0] - old_vel[0])),
                        old_vel[1] + max(-self.acc_limit.vel[1],
                                         min(self.acc_limit.vel[1], new_vel[1] - old_vel[1])),
                        old_vel[2] + max(-self.acc_limit.rot[2],
                                         min(self.acc_limit.rot[2], new_vel[2] - old_vel[2]))])
        return vel

    def send_empty_twist(self):
        self.cmd_vel_sub.publish(Twist())

    def execute_cb(self, data):
        """
        :type data: FollowJointTrajectoryGoal
        :return:
        """
        try:
            x_index = data.trajectory.joint_names.index(x_joint)
            y_index = data.trajectory.joint_names.index(y_joint)
            z_index = data.trajectory.joint_names.index(z_joint)
            time_tolerance = 0.1
            self.send_empty_twist()
            self.start_time = data.trajectory.header.stamp.to_sec()
            self.pose_traj2, self.goal_vel2, self.time_traj = traj_to_poses(data, x_index, y_index, z_index)
            self.current_index = 0
            self.done = False
            while self.current_index < len(self.time_traj) - 4 and not self.server.is_preempt_requested():
                current_time = self.time_traj[self.current_index]
                tfs = rospy.get_rostime().to_sec() - self.start_time
                rospy.sleep(current_time - tfs)
                self.current_index += 1
            self.done = True
            rospy.sleep(time_tolerance)
            self.cmd = None
            rospy.loginfo(
                'goal reached, final diff: {}'.format(self.pose_traj2(self.time_traj[-1]) - self.current_pose))
            self.current_goal = None
            self.start_time = None
            self.server.set_succeeded()
        except:
            traceback.print_exc()
            rospy.loginfo('aborted current goal')
            self.server.set_aborted()
        finally:
            self.current_goal = None
            self.cmd_vel_sub.publish(Twist())


if __name__ == '__main__':
    try:
        rospy.init_node('omni_pose_follower')
        name_space = rospy.get_param('~name_space')
        vel_tolerance = rospy.get_param('~vel_tolerance')
        x_joint = rospy.get_param('{}/odom_x_joint'.format(name_space))
        y_joint = rospy.get_param('{}/odom_y_joint'.format(name_space))
        z_joint = rospy.get_param('{}/odom_z_joint'.format(name_space))
        opf = OmniPoseFollower()
        rospy.loginfo('pose follower running')
        rospy.spin()

    except rospy.ROSInterruptException:
        pass
