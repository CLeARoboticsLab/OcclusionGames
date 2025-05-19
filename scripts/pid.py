#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Point, Twist
from nav_msgs.msg import Odometry

X_GOAL = 50
ERR_EPSILON = 0.1

### initially, this will be just for moving 50m ahead. ###
class PID_Runner:
    def __init__(self):
        self.Kp = 0.25
        self.goal = Point()
        self.goal.x = X_GOAL
        self.run = True
        self.err = X_GOAL
        self.speed = Twist()
        self.cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        self.odom_sub = rospy.Subscriber("/odom", Odometry, self.update)

    def update(self, msg):
        # first, update the error
        curr_pos = msg.pose.pose.position
        self.update_err(curr_pos)
        # then, update the speed
        self.speed.linear.x = self.Kp * self.err
        self.cmd_vel_pub.publish(self.speed)
        # finally, check whether we should continue
        if (abs(self.err) <= ERR_EPSILON):
            # reached close enough to goal, stop everything
            self.run = False
            self.speed.linear.x = 0            

    def update_err(self, curr_pos):
        self.err = self.goal.x - curr_pos.x

if __name__ == "__main__":
    rospy.init_node("racecar_pid_runner", anonymous=True)
    pid_runner = PID_Runner()
    while pid_runner.run: # need to keep advancing toward goal
        pass
    rospy.spin() # shutdown this node
