#!/usr/bin/env python3

"""
Author: Jai Nagaraj
This Python scipt sets up a TCP server for a JetRacer to receive steering and throttle data,
as well as send out pose data for the JetRacer.
Data is expected to be sent as: {"throttle": <float>, "steering": <float>}
Data from this server is sent as: {"x": <float>, "y": <float>, "v": <float>, "psi": <float>}

NOTE: Data is only sent when the message "get_pose" is received.

"""

import socket
import threading
import json
import rospy
from std_msgs.msg import Float32
from geometry_msgs.msg import PoseStamped, TwistStamped
import math
import time
import signal
import sys

HOST = '0.0.0.0'  # localhost
PORT = 65432        # non-privileged port
SEND_POSE_CODE = "get_pose"
STOP_CODE = "STOP"

POSE_TOPIC = "vrpn_client_node/JaiAliJetRacer/pose"
TWIST_TOPIC = "vrpn_client_node/JaiAliJetRacer/twist"
THROTTLE_TOPIC = "/jetracer/throttle"
STEERING_TOPIC = "/jetracer/steering"

throttle_pub = None
steering_pub = None
pose_sub = None
latest_state = [None,None,0,None]
LATEST_X = 0
LATEST_Y = 1
LATEST_V = 2
LATEST_PSI = 3

server_socket = None  # Global reference for cleanup

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

def pose_callback(data: PoseStamped):
    # update latest state stuff (except velocity)
    latest_state[LATEST_X] = data.pose.position.x
    latest_state[LATEST_Y] = data.pose.position.y
    _, _, latest_state[LATEST_PSI] = euler_from_quaternion(
        data.pose.orientation.x,
        data.pose.orientation.y,
        data.pose.orientation.z,
        data.pose.orientation.w
    )

def twist_callback(data: TwistStamped):
    v_x = data.twist.linear.x
    v_y = data.twist.linear.y
    latest_state[LATEST_V] = math.hypot(v_x, v_y)

def handle_receive(conn):
    while True:
        data = conn.recv(1024)
        if not data:
            print("No data received, closing connection...")
            break
        decoded_data = data.decode("utf-8")
        if decoded_data == SEND_POSE_CODE:
            # requesting latest pose data
            while (latest_state[LATEST_X] is None):
                print("waiting...")
                time.sleep(0.5) # waits for valid pose data to arrive
            dict_to_send = {
                "x": latest_state[LATEST_X],
                "y": latest_state[LATEST_Y],
                "v": latest_state[LATEST_V],
                "psi": latest_state[LATEST_PSI]
            }
            json_str_to_send = json.dumps(dict_to_send)
            conn.sendall(json_str_to_send.encode())
        elif decoded_data == STOP_CODE:
            # Stop the subscribers and close the connection
            print("Closing connection...")
            conn.close()
            print("Connection closed.")
            break
        else:
            # contains JSON with controls
            json_obj = json.loads(decoded_data)
            print(f"Throttle received: {json_obj['throttle']}")
            print(f"Steering received: {json_obj['steering']}\n")
            # output throttle, steering to jetracer
            throttle_pub.publish(Float32(float(json_obj['throttle'])))
            steering_pub.publish(Float32(float(json_obj['steering'])))
    print("Exiting receive handler...")

def sigint_handler(signum, frame):
    print("\nSIGINT received, shutting down server...")
    if server_socket:
        try:
            server_socket.close()
            print("Server socket closed.")
        except Exception as e:
            print(f"Error closing server socket: {e}")
    rospy.signal_shutdown("SIGINT received")
    sys.exit(0)

def main():
    global throttle_pub, steering_pub, pose_sub, server_socket
    rospy.init_node("tcp_jetracer_server")
    throttle_pub = rospy.Publisher(THROTTLE_TOPIC, Float32, queue_size=1)
    steering_pub = rospy.Publisher(STEERING_TOPIC, Float32, queue_size=1)
    pose_sub = rospy.Subscriber(POSE_TOPIC, PoseStamped, pose_callback)
    #twist_sub = rospy.Subscriber(TWIST_TOPIC, TwistStamped, twist_callback)

    # Register SIGINT handler
    signal.signal(signal.SIGINT, sigint_handler)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        server_socket = s  # Assign to global for cleanup
        s.bind((HOST, PORT))
        s.listen()
        print(f"Server listening on {HOST}:{PORT}...")
        while not rospy.is_shutdown():
            print("Waiting for a new connection...")
            try:
                conn, addr = s.accept()
            except OSError:
                # Socket closed, exit loop
                break
            print(f"Connected by {addr}")
            threading.Thread(target=handle_receive, args=(conn,), daemon=True).start()
    rospy.spin()

if __name__ == "__main__":
    main()
