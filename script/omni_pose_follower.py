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
    return PyKDL.Frame(PyKDL.Rotation.Quaternion(pose.orientation.x,
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


class OmniPoseFollower(object):
    def __init__(self):
        urdf = rospy.get_param('robot_description')
        self.urdf = URDF.from_xml_string(hacky_urdf_parser_fix(urdf))
        self.x_limit = self.urdf.joint_map[x_joint].limit.velocity
        self.y_limit = self.urdf.joint_map[y_joint].limit.velocity
        self.z_limit = self.urdf.joint_map[z_joint].limit.velocity
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

        goal = self.current_goal
        if goal:
            error = PyKDL.diff(self.current_pose, goal)
            cmd = error * p - (error - self.last_error) * d
            cmd = self.limit_vel(cmd)
            self.last_error = cmd

            cmd = kdl_to_twist(cmd)
            self.cmd_vel_sub.publish(cmd)

        state = JointTrajectoryControllerState()
        state.joint_names = [x_joint, y_joint, z_joint]
        self.state_pub.publish(state)

    def limit_vel(self, vel):
        """
        :type vel: PyKDL.Twist
        :return:
        """
        vel.vel[0] = min(self.x_limit, max(-self.x_limit, vel.vel[0]))
        vel.vel[1] = min(self.y_limit, max(-self.y_limit, vel.vel[1]))
        vel.rot[2] = min(self.z_limit, max(-self.z_limit, vel.rot[2]))
        return vel

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
            i = 1
            time_tolerance = 0.1
            while i < len(data.trajectory.points) and not self.server.is_preempt_requested():
                current_point = data.trajectory.points[i]
                time_from_start = rospy.get_rostime().to_sec() - self.start_time
                if time_from_start < current_point.time_from_start.to_sec():
                    self.current_goal = self.js_to_kdl(current_point.positions[x_index],
                                                       current_point.positions[y_index],
                                                       current_point.positions[z_index])
                else:
                    i += 1
                tfs = rospy.get_rostime().to_sec() - self.start_time
                asdf = current_point.time_from_start.to_sec() - tfs
                rospy.sleep(asdf)
            rospy.sleep(time_tolerance)
            rospy.loginfo('goal reached, final diff: {}'.format(PyKDL.diff(self.current_pose, self.current_goal)))
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
        x_joint = rospy.get_param('{}/odom_x_joint'.format(name_space))
        y_joint = rospy.get_param('{}/odom_y_joint'.format(name_space))
        z_joint = rospy.get_param('{}/odom_z_joint'.format(name_space))
        p = rospy.get_param('{}/p'.format(name_space))
        d = rospy.get_param('{}/d'.format(name_space))
        opf = OmniPoseFollower()
        rospy.loginfo('pose follower running')
        rospy.spin()

    except rospy.ROSInterruptException:
        pass
