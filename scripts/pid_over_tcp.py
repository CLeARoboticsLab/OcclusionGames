#!/usr/bin/env python3

"""
Authors: Jai Nagaraj, Ali Chimoun
This script is designed to execute PID Control over a TCP connection to an NVIDIA JetRacer.
Data is expected to be sent as: {"x": <float>, "y": <float>, "v": <float>, "psi": <float>}
Data from this server is sent as: {"throttle": <float>, "steering": <float>}
"""

import math
import time
import sys
import socket
import json

ERR_EPSILON = 0.2
GOAL = [-3, 3]
THETA = 0
HEADING_BIAS = 0.04

HOST = "192.168.50.236"
PORT = 65432        # non-privileged port
RECV_CODE = "get_pose"

X_ARG = 1
Y_ARG = 2
THETA_ARG = 3

class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.integral = 0
        self.last_error = 0
        self.last_time = time.time()

    def reset(self):
        self.integral = 0
        self.last_error = 0
        self.last_time = time.time()

    def update(self, error):
        current_time = time.time()
        dt = current_time - self.last_time
        de = error - self.last_error

        self.integral += error * dt
        derivative = de / dt if dt > 0 else 0

        output = self.kp * error + self.ki * self.integral + self.kd * derivative

        self.last_error = error
        self.last_time = current_time

        return output

class JetRacerController:
    def __init__(self, conn):
        self.goal = GOAL  # example goal point in meters

        self.heading_pid = PIDController(0.4, 0.0, 0.04)
        self.distance_pid = PIDController(0.5, 0.0, 0.35) # Try pure proportinality first
        self.conn = conn

    def control(self):
        # Infinite loop to get to goal
        while True:
            # first, get pose information
            self.conn.sendall(RECV_CODE.encode("utf-8"))
            raw_pose_data = self.conn.recv(1024).decode()
            pose_dict = json.loads(raw_pose_data)
            x = pose_dict['x']
            y = pose_dict['y']
            print('Postion:', x, y)
            yaw = pose_dict['psi']
            print('Heading:', yaw)

            dx = self.goal[0] - x
            dy = self.goal[1] - y
            goal_dist = math.hypot(dx, dy)
            goal_heading = math.atan2(dy, dx)
            heading_error = self.angle_wrap(goal_heading - yaw)

            # Compute control
            steering = self.heading_pid.update(heading_error)
            throttle = self.distance_pid.update(goal_dist)

            # Limit values
            steering = max(min(steering + HEADING_BIAS, 1.0), -1.0)
            throttle = max(min(throttle, 0.2), 0.0) if goal_dist > ERR_EPSILON else 0.0

            # Send values over TCP
            print("Throttle sent:", throttle)
            print("Steering sent:", steering, '\n')
            control_dict = {'throttle': throttle, 'steering': steering}
            control_str = json.dumps(control_dict) # converts dict to json string
            self.conn.sendall(control_str.encode("utf-8"))
            #time.sleep(0.01) # delay so that we don't bombard the server
        

    def angle_wrap(self, angle):
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

def main(argc, argv):
    #if argc != 3:
    #    print("USAGE: python3 pid_official.py [x_coordinate] [y_coordinate] [theta]")
    #    return
    #if argc != 2:
    #    print("USAGE: python3 pid_official.py [x_coordinate] [y_coordinate]")
    #    return
    #GOAL[0] = argv[X_ARG]
    #GOAL[1] = argv[Y_ARG]
    #THETA = kwargs[THETA_ARG]

    # connect to Jetson TCP server
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
        client_socket.connect((HOST, PORT))
        controller = JetRacerController(client_socket)
        controller.control()

if __name__ == '__main__':
    main(len(sys.argv), sys.argv)
