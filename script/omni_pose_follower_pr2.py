#!/usr/bin/env python
import numpy as np
import traceback

import PyKDL
import rospy
from actionlib import SimpleActionServer
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal, JointTrajectoryControllerState
from geometry_msgs.msg import Twist
from scipy.interpolate import splrep, splev
from sensor_msgs.msg import JointState
from tf.transformations import unit_vector
from urdf_parser_py.urdf import URDF

# TODO possible improvements
# spline interpolation on original traj
# limit

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


def make_cmd2(current_vel, vel_traj, current_pose, pos_traj, pose_history, current_time, time_traj, i, eps=1e-5):
    goal_pose = pos_traj[i]
    goal_vel = vel_traj[i]
    pose_error = goal_pose - current_pose
    # dt = time_traj[i] - current_time
    # error_vel = pose_error / dt
    if np.linalg.norm(current_vel) == 0:
        # current_vel = goal_vel
        return goal_vel
    else:
        goal_vel2 = vel_traj[i - 1]
        reference_angle = angular_distance(goal_vel2, goal_vel)
        needed_angle = angular_distance(current_vel, pose_error)
        beta = min(needed_angle, reference_angle + 0.01)
        cmd = interpolate(current_vel, pose_error, beta)
        if np.linalg.norm(pose_error) < eps:
            return np.array([0, 0, 0])
        cmd = cmd * max(np.linalg.norm(goal_vel), min(np.linalg.norm(pose_error), 0.1))
        return cmd


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

def limitTwist(twist, min_vel_lin_=0.005, max_vel_lin_=0.142, min_vel_th_=0.005, max_vel_th_=0.2, min_in_place_vel_th_=0.0,
               in_place_trans_vel_=0.0):
    res = twist
    # make sure to bound things by our velocity limits
    lin_overshoot = np.sqrt(res.linear.x * res.linear.x + res.linear.y * res.linear.y) / max_vel_lin_
    lin_undershoot = min_vel_lin_ / np.sqrt(res.linear.x * res.linear.x + res.linear.y * res.linear.y)
    if lin_overshoot > 1.0:
        res.linear.x /= lin_overshoot
        res.linear.y /= lin_overshoot
        # keep relations
        res.angular.z /= lin_overshoot

    # we only want to enforce a minimum velocity if we're not rotating in place
    if lin_undershoot > 1.0:
        res.linear.x *= lin_undershoot
        res.linear.y *= lin_undershoot
        # we cannot keep relations here for stability reasons

    if np.fabs(res.angular.z) > max_vel_th_:
        scale = max_vel_th_ / np.fabs(res.angular.z)
        # res.angular.z = max_vel_th_ * sign(res.angular.z);
        res.angular.z *= scale
        # keep relations
        res.linear.x *= scale
        res.linear.y *= scale

    if np.fabs(res.angular.z) < min_vel_th_:
        res.angular.z = min_vel_th_ * np.sign(res.angular.z)
    # we cannot keep relations here for stability reasons

    # we want to check for whether or not we're desired to rotate in place
    if np.sqrt(twist.linear.x * twist.linear.x + twist.linear.y * twist.linear.y) < in_place_trans_vel_:
        if np.fabs(res.angular.z) < min_in_place_vel_th_:
            res.angular.z = min_in_place_vel_th_ * np.sign(res.angular.z)
            print('rotate')
        print('dont translate')
        res.linear.x = 0.0
        res.linear.y = 0.0

    return res


def make_cmd(current_vel, vel_traj, current_pose, pos_traj, pose_history, current_time, time_traj, i):
    goal_pose = pos_traj[i]
    goal_vel = vel_traj[i]
    pose_error = goal_pose - current_pose

    time = np.array([current_time,
                     time_traj[i],
                     time_traj[i + 1],
                     time_traj[i + 2],
                     time_traj[i + 3]])
    y = np.array([current_pose,
                  pos_traj[i],
                  pos_traj[i + 1],
                  pos_traj[i + 2],
                  pos_traj[i + 3]])
    tck_x = splrep(time, y[:, 0], s=0)
    tck_y = splrep(time, y[:, 1], s=0)
    tck_z = splrep(time, y[:, 2], s=0)
    # xnew = np.array([current_time])
    xnew = np.arange(time[0], time[-1], 0.01)
    ynew_x = splev(xnew, tck_x, der=0)
    ynew_y = splev(xnew, tck_y, der=0)
    ynew_y_d = splev(xnew, tck_y, der=1)
    ynew_z = splev(xnew, tck_z, der=0)
    dt = (xnew[10] - xnew[0])
    cmd = np.array([(ynew_x[10] - ynew_x[0]) / dt,
                    (ynew_y[10] - ynew_y[0]) / dt,
                    (ynew_z[10] - ynew_z[0]) / dt])
    return cmd

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
    return position_traj, vel_traj, time_traj


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
        current_vel = np.array([js.velocity[self.x_index_js],
                                js.velocity[self.y_index_js],
                                js.velocity[self.z_index_js]])
        # current_vel = PyKDL.Twist()
        # current_vel.vel[0] = js.velocity[self.x_index_js]
        # current_vel.vel[1] = js.velocity[self.y_index_js]
        # current_vel.rot[2] = js.velocity[self.z_index_js]
        time = js.header.stamp.to_sec()
        # if self.pose_history is None:
        #     self.pose_history = deque([[0, 0, 0], np.array([self.current_pose, current_pose, current_pose])], 3)
        # else:
        #     self.pose_history.append([current_pose])
        if not self.done:
            # if self.cmd and rospy.get_rostime().to_sec() > self.start_time:
            time_from_start = time - self.start_time
            if time_from_start > 0:
                # if sim:
                #     cmd = self.cmd
                #     # cmd_msg = kdl_to_twist(self.cmd)
                # else:
                #     cmd = self.current_pose.M.Inverse() * self.cmd

                # if self.done:
                #     error = (self.current_goal - self.current_pose)
                # else:
                #     error = self.cmd

                # cmd = np_to_twist(interpolate(current_vel, error))
                # cmd = scale_cmd(interpolated_cmd, np.linalg.norm(kdl_to_np(error)))
                # cmd = self.limit_vel(cmd)
                cmd = make_cmd2(current_vel, self.vel_traj, self.current_pose, self.pose_traj, self.pose_history,
                                time_from_start, self.time_traj, self.current_index)
                if np.any(np.isnan(cmd)):
                    print('fuck')
                else:
                    # cmd = self.limit_acceleration(current_vel, cmd)
                    cmd_msg = np_to_msg(cmd)

                    # dt = self.time_traj[self.current_index] - time_from_start
                    # error = self.pose_traj[self.current_index] - current_pose
                    # error_dt = error / dt
                    # cmd_msg = limitTwist(cmd_msg)
                    self.cmd_vel_sub.publish(cmd_msg)

        state = JointTrajectoryControllerState()
        state.joint_names = [x_joint, y_joint, z_joint]
        self.state_pub.publish(state)

    def limit_correction(self, x, ref):
        if x > 0:
            return max(min(x, ref + vel_tolerance), ref - vel_tolerance)
        return min(max(x, ref - vel_tolerance), ref + vel_tolerance)

    def limit_vel(self, twist):
        twist.vel[0] = max(min(twist.vel[0], self._max_vel.vel[0]), self._min_vel.vel[0])
        twist.vel[1] = max(min(twist.vel[1], self._max_vel.vel[1]), self._min_vel.vel[1])
        twist.rot[2] = max(min(twist.rot[2], self._max_vel.rot[2]), self._min_vel.rot[2])
        return twist

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
            self.done = False
            self.last_error = PyKDL.Twist()
            x_index = data.trajectory.joint_names.index(x_joint)
            y_index = data.trajectory.joint_names.index(y_joint)
            z_index = data.trajectory.joint_names.index(z_joint)
            self.current_index = 0
            self.hz = data.trajectory.points[1].time_from_start.to_sec() - \
                      data.trajectory.points[0].time_from_start.to_sec()
            time_tolerance = 0.1
            last_stamp = 0
            self.pose_history = None
            self.send_empty_twist()
            self.start_time = data.trajectory.header.stamp.to_sec()
            self.pose_traj, self.vel_traj, self.time_traj = traj_to_poses(data, x_index, y_index, z_index)
            while self.current_index < len(self.time_traj) - 4 and not self.server.is_preempt_requested():
                # current_point = data.trajectory.points[self.current_index]
                current_time = self.time_traj[self.current_index]
                # time_from_start = rospy.get_rostime().to_sec() - self.start_time
                # time_from_start2 = current_point.time_from_start.to_sec()
                # self.current_goal = self.js_to_kdl(current_point.positions[x_index],
                #                                    current_point.positions[y_index],
                #                                    current_point.positions[z_index])
                # self.current_goal_vel = PyKDL.Twist()
                # self.current_goal_vel.vel[0] = current_point.velocities[x_index]
                # self.current_goal_vel.vel[1] = current_point.velocities[y_index]
                # self.current_goal_vel.rot[2] = current_point.velocities[z_index]

                # self.dt = time_from_start2 - last_stamp
                # last_stamp = time_from_start2

                # self.error = (self.current_goal - self.current_pose) / dt

                # cmd = self.error / dt
                # cmd.vel[0] = self.limit_correction(cmd.vel[0], self.current_goal_vel.vel[0])
                # cmd.vel[1] = self.limit_correction(cmd.vel[1], self.current_goal_vel.vel[1])
                # cmd.rot[2] = self.limit_correction(cmd.rot[2], self.current_goal_vel.rot[2])
                # self.cmd = cmd
                tfs = rospy.get_rostime().to_sec() - self.start_time
                # asdf = current_point.time_from_start.to_sec() - tfs
                asdf = current_time - tfs
                rospy.sleep(asdf)
                self.current_index += 1
            self.done = True
            rospy.sleep(time_tolerance)
            self.cmd = None
            rospy.loginfo('goal reached, final diff: {}'.format(self.pose_traj[-1] - self.current_pose))
            print('done')
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
