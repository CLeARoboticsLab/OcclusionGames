#!/usr/bin/env python3
import rospy
import csv
from std_msgs.msg import Float32
from geometry_msgs.msg import PoseStamped, TwistStamped, AccelStamped
import time
import math

POSE_TOPIC = "/vrpn_client_node/JaiAliJetRacer/pose"
VEL_TOPIC = "/vrpn_client_node/JaiAliJetRacer/twist"
ACCEL_TOPIC = "/vrpn_client_node/JaiAliJetRacer/accel" # Remind me of Accel World lol


class ViconLogger:
    def __init__(self):
        self.logfile = open("data_log.csv", "w")
        self.writer = csv.writer(self.logfile)
        self.writer.writerow(["Time", "x", "y", "z", "yaw", "yaw_rate", "vx", "vy", "vz","velocity" ,"ax", "ay", "az", "acceleration", "throttle", "steering"])

        self.throttle = 0.0
        self.steering = 0.0
        self.pose = None
        self.accel = None

        rospy.Subscriber(VEL_TOPIC, TwistStamped, self.vicon_callback) # I'll start by looking directly at the velocity provided by the vicon
        # If it's garbage, we'll have to resort to some sort of finite difference scheme with the position
        rospy.Subscriber(POSE_TOPIC, PoseStamped, self.pose_callback)
        #rospy.Subscriber(ACCEL_TOPIC, AccelStamped, self.accel_callback)
        rospy.Subscriber("/jetracer/throttle", Float32, self.throttle_callback)
        rospy.Subscriber("/jetracer/steering", Float32, self.steering_callback)

    def vicon_callback(self, msg):

        linear = msg.linear
        angular = msg.angular
        
        p = self.pose.position # Coordinates
        q = self.pose.orientation # Angular position
        
        a = self.accel.linear

        velocity = math.hypot(linear.x, linear.y)
        acceleration = math.hypot(a.x, a.y)

        now = rospy.get_time()

        _, _, yaw = euler_from_quaternion(q.x, q.y, q.z, q.w)


        print("I'm writing something")
        self.writer.writerow([now, p.x, p.y, p.z, yaw, angular.z, linear.x, linear.y, linear.z, velocity, a.x, a.y, a.z, acceleration, self.throttle, self.steering])
        self.logfile.flush()


    def pose_callback(self, msg):
        self.pose = msg

    def accel_callback(self, msg):
        self.accel = msg

    def throttle_callback(self, msg):
        self.throttle = msg.data

    def steering_callback(self, msg):
        self.steering = msg.data


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
     
        return roll_x, pitch_y, yaw_z # in radians


if __name__ == '__main__':
    rospy.init_node('vicon_logger')
    logger = ViconLogger()
    rospy.spin()
    logger.logfile.close()
