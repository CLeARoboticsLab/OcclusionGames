#!/usr/bin/env python3
import rospy
import os
import csv
from std_msgs.msg import Float32
from geometry_msgs.msg import PoseStamped, TwistStamped, AccelStamped
import time
import math

POSE_TOPIC = "/vrpn_client_node/JaiAliJetRacer/pose"
VEL_TOPIC = "/vrpn_client_node/JaiAliJetRacer/twist"
#ACCEL_TOPIC = "/vrpn_client_node/JaiAliJetRacer/accel" # Remind me of Accel World lol
THROTTLE_VALUE = 0.2
SINE_AMP = 1
SINE_FREQ = 6
X_STOP = 3.0
T_STOP = 10
HEADING_BIAS = 0.04


class ViconLogger:
    def __init__(self):
        #print(os.getcwd())
        self.logfile = open("dynamics_csv/faster_sine_dynamics_fernando.csv", "w")
        self.writer = csv.writer(self.logfile)
        self.writer.writerow(["Time", "steering", "throttle", "x", "y", "vx", "vy", "velocity", "yaw", "yaw_rate"])
        init_pose = rospy.wait_for_message(POSE_TOPIC, PoseStamped)
        self.position = init_pose.pose
        self.start_time = rospy.get_time()

        self.throttle_pub = rospy.Publisher("/jetracer/throttle", Float32, queue_size=1)
        self.steering_pub = rospy.Publisher("/jetracer/steering", Float32, queue_size=1)
        #self.throttle_pub.publish(Float32(THROTTLE_VALUE))
        #self.steering_pub.publish(Float32(STEERING_VALUE))
        self.vel_sub = rospy.Subscriber(VEL_TOPIC, TwistStamped, self.vicon_callback) # I'll start by looking directly at the velocity provided by the vicon
        # If it's garbage, we'll have to resort to some sort of finite difference scheme with the position
        self.pose_sub = rospy.Subscriber(POSE_TOPIC, PoseStamped, self.pose_callback)

    def vicon_callback(self, msg):

        msg = msg.twist
        linear = msg.linear
        angular = msg.angular
        
        p = self.position.position # Coordinates
        q = self.position.orientation # Angular position
        
        #a = self.acceleration.linear
        
        velocity = math.hypot(linear.x, linear.y)
        #acceleration = math.hypot(a.x, a.y)

        now = rospy.get_time()

        _, _, yaw = euler_from_quaternion(q.x, q.y, q.z, q.w)

        # calculate steering value
        steering_value = SINE_AMP * math.sin(SINE_FREQ * (now - self.start_time)) + HEADING_BIAS
        steering_value = max(-1, min(1, steering_value)) # clamp


        self.writer.writerow([now - self.start_time, steering_value, THROTTLE_VALUE, p.x, p.y, linear.x, linear.y, velocity, yaw, angular.z])
        self.logfile.flush()
        self.throttle_pub.publish(Float32(THROTTLE_VALUE))
        self.steering_pub.publish(Float32(steering_value))


    def pose_callback(self, msg):
        self.position = msg.pose
        t = rospy.get_time()
        if self.position.position.x >= X_STOP or t - self.start_time > T_STOP:
            self.vel_sub.unregister()
            self.pose_sub.unregister()
            self.throttle_pub.publish(0)
            self.steering_pub.publish(0)
            print("ALL DONE!")

    #def accel_callback(self, msg):
        #self.acceleration = msg.accel

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
