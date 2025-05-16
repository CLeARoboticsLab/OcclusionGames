#!/usr/bin/env python3

import rospy
from geomtry_msgs.msg import Point, Twist
from nav_msgs.msg import Odometry

class PID_Runner:
    def __init__(self):
        self.goal = Point()
        #TODO: edit goal
        self.run = True
        self.err = 0
        self.speed = Twist()
        self.cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        self.odom_sub = rospy.Subscriber("/odom", Odometry, self.update_err)
    def calc_err(self):
        pass
