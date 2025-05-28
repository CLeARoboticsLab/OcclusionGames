#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Point, Twist, PoseStamped
from std_msgs.msg import Float32
import time
import math
import numpy as np

X_GOAL = 0
Y_GOAL = 5
TARGET = np.array([X_GOAL, Y_GOAL])
ERR_EPSILON = 0.01
MAX_SPEED = 15 # in m/s
MIN_SPEED = 0.05
EXEC_RATE = 10
POSE_TOPIC = "vrpn_client_node/JaiAliJetRacer2/pose"

### initially, this will be just for moving 10m ahead. ###
class PID_Runner:
    def __init__(self):
        self.start_pose = rospy.wait_for_message(POSE_TOPIC, PoseStamped, timeout=2)
        self.start_pose = self.start_pose.pose
        self.start_time = time.time()
        self.t = 0
        self.sleep_time = 0
        self.x = self.start_pose.position.x
        self.y = self.start_pose.position.y
        self.orient = self.start_pose.orientation
        _,_,self.yaw = euler_from_quaternion(self.orient.x, self.orient.y, self.orient.z, self.orient.w)
        self.Kp = 0.25
        self.Ki = 0.1
        self.Kd = 0.2
        self.goal = Point()
        self.goal.x = self.x + X_GOAL
        self.goal.y = self.y + Y_GOAL
        self.run = True
        self.correctX = self.x
        self.correctY = self.y
        self.xerr = 0
        self.yerr = 1
        self.speed = 0
        self.pose_sub = rospy.Subscriber(POSE_TOPIC, PoseStamped, self.update)
        self.throttle_pub = rospy.Publisher("/jetracer/throttle", Float32, queue_size=1)
        self.steering_pub = rospy.Publisher("/jetracer/steering", Float32, queue_size=1)

    def update(self, msg):
        msg = msg.pose
        print("Speed: ", self.speed)
        #print("Time elapsed: ", self.t)
        print("Y error: ", self.yerr)
        print("Correct Y: ", self.correctY)
        print("Current Y: ", self.y, '\n')
        # first, update the current position and error
        curr_time = time.time()
        self.t = curr_time - self.start_time
        self.x = msg.position.x
        self.y = msg.position.y
        self.update_err(msg)
        # then, update the speed and the turn angle
        # Speed
        self.speed = math.tanh(self.Kp * (self.xerr + self.yerr)) / 6 
        #self.speed = self.yerr
        self.turn = math.tanh(self.Kp * self.yawerr) / 4.0
        self.throttle_pub.publish(self.speed)
        # finally, check whether we should continue
        if abs(self.x - self.goal.x) <= ERR_EPSILON and abs(self.y - self.goal.y) <= ERR_EPSILON:
            # reached close enough to goal, stop everything
            self.run = False
            self.speed = 0
            self.throttle_pub.publish(self.speed)
            self.pose_sub.unregister()

        # Turn angle
        self.steering_pub.publish(self.turn)



    def update_err(self, msg):
        self.correctX = self.start_pose.position.x + 0*self.t
        self.xerr = self.correctX - self.x
        self.correctY = self.start_pose.position.y + 1.5*self.t
        self.yerr = self.correctY - self.y
        _, _, curr_yaw = euler_from_quaternion(msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w)

        # Expressing the position of the target as viewed from our position
        # (Vector subtraction)
        pose_vec = np.array([msg.position.x, msg.position.y])
        target = TARGET - pose_vec
        self.correct_yaw = np.arccos(target[1] / np.linalg.norm(target))
        self.yawerr = self.correct_yaw  - curr_yaw

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

if __name__ == "__main__":
    rospy.init_node("racecar_pid_runner", anonymous=True)
    pid_runner = PID_Runner()
    rate = rospy.Rate(EXEC_RATE)
    while pid_runner.run:
        pass
    print('Done')
    rospy.spin() # shutdown this node
