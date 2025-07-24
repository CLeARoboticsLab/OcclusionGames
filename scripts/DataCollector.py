import rospy
import math
from geometry_msgs.msg import PoseStamped, TwistStamped

"""
A simple data collector for the JetRacer, subscribing to pose and twist topics.
It stores the latest pose and twist data, which can be accessed by other components.
"""
class DataCollector:
    def __init__(self, first=True):
        # initialize subscribers
        if first:
            self.pose_sub = rospy.Subscriber('/vrpn_client_node/JaiAliJetRacer/pose', PoseStamped, self.pose_callback)
            self.twist_sub = rospy.Subscriber('/vrpn_client_node/JaiAliJetRacer/twist', TwistStamped, self.twist_callback)
        else:
            self.pose_sub = rospy.Subscriber('/vrpn_client_node/JaiAliJetRacerTwo/pose', PoseStamped, self.pose_callback)
            self.twist_sub = rospy.Subscriber('/vrpn_client_node/JaiAliJetRacerTwo/twist', TwistStamped, self.twist_callback)
        # initialize states
        self.latest_pose = [0, 0, 0] # [x, y, psi]
        self.latest_twist = [0, 0, 0] # [vx, vy, psi_dot]

    def pose_callback(self, msg):
        _, _, psi = self.euler_from_quaternion(
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w
        )
        self.latest_pose = [msg.pose.position.x, msg.pose.position.y, psi]

    def twist_callback(self, msg):
        self.latest_twist = [msg.twist.linear.x, msg.twist.linear.y, msg.twist.angular.z]

    def euler_from_quaternion(x, y, z, w):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
        
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
        
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)

        return roll_x, pitch_y, yaw_z  # in radians