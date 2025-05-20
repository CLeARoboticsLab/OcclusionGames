#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Point, Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32
import time
import math

X_GOAL = 7
Y_GOAL = 5
ERR_EPSILON = 0.01
MAX_SPEED = 15 # in m/s
MIN_SPEED = 0.05
EXEC_RATE = 10

### initially, this will be just for moving 10m ahead. ###
class PID_Runner:
    def __init__(self):
        self.curr_time = time.time()
        self.sleep_time = 0
        self.distance = 0
        self.Kp = 0.25
        self.goal = Point()
        self.goal.x = X_GOAL
        self.run = True
        self.err = X_GOAL
        self.speed = 0
        self.throttle_pub = rospy.Publisher("/jetracer/throttle", Float32, queue_size=1)
        self.steering_pub = rospy.Publisher("/jetracer/steering", Float32, queue_size=1)

    def update(self):
        print("Speed: ", self.speed)
        print("Distance traveled so far:", self.distance)
        # first, update the error
        curr_time = time.time()
        elapsed_time = curr_time - self.curr_time - self.sleep_time
        self.curr_time = curr_time
        self.distance = self.distance + MAX_SPEED * self.speed * elapsed_time
        self.update_err(self.distance)
        # then, update the speed
        self.speed = math.tanh(self.Kp * self.err) / 4.0
        if abs(self.speed) < MIN_SPEED:
            self.speed = -MIN_SPEED if self.speed < 0.0 else MIN_SPEED
        self.throttle_pub.publish(self.speed)
        # finally, check whether we should continue
        if (abs(self.err) <= ERR_EPSILON):
            # reached close enough to goal, stop everything
            self.run = False
            self.speed = 0
            self.throttle_pub.publish(self.speed)

    def run(self):
        throttle_pub.publish(0.1)
        go(0)
        turn90()
        go(1)
        turn90()
        go(0)
        turn90()
        go(1)
        turn90()
        throttle_pub.publish(0)

    def turn90(self):
        turn_time = 0.6
        self.steering_pub(1)
        start_time = time.time()
        while True:
            if (time.time() - start_time > turn_time):
                break
        self.steering_pub.publish(0)
    
    def go(self, side):
        distance = 0
        if side == 0:
            # short side
            distance = 2
        else:
            #long side
            distance = 8
        # calculate time to travel the distance
        time_req = distance / (MAX_SPEED * 0.1)
        start_time = time.time()
        while True:
            if (time.time() - start_time > time_req):
                break

if __name__ == "__main__":
    rospy.init_node("racecar_pid_runner", anonymous=True)
    pid_runner = PID_Runner()
    #rate = rospy.Rate(EXEC_RATE)
    #while pid_runner.run: # need to keep advancing toward goal
    #    pid_runner.update()
    #    st = time.time()
    #    rate.sleep()
    #    end = time.time()
    #    pid_runner.sleep_time = (end - st) / 4
    pid_runner.run()
    rospy.spin() # shutdown this node
