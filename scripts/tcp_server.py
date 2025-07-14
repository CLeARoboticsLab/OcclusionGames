#!/usr/bin/env python3

"""
Author: Jai Nagaraj
This Python scipt sets up a TCP server for a JetRacer to receive steering and throttle data.
Data is expected to be sent as: {"throttle": <float>, "steering": <float>}
"""

import socket
import threading
import json
import rospy
from std_msgs import Float32
from geometry_msgs import PoseStamped
import time

HOST = '0.0.0.0'  # localhost
PORT = 65432        # non-privileged port

POSE_TOPIC = "vrpn_client_node/JaiAliJetRacer/pose"
THROTTLE_TOPIC = "/jetracer/throttle"
STEERING_TOPIC = "/jetracer/steering"

throttle_pub = None
steering_pub = None
pose_sub = None

def handle_receive(conn):
    while True:
        data = conn.recv(1024)
        if not data:
            break
        decoded_data = data.decode("utf-8")
        json_obj = json.loads(decoded_data)
        print(f"Throttle received: {json_obj["throttle"]}")
        print(f"Throttle received: {json_obj["steering"]}\n")

        # output throttle, steering to jetracer
        throttle_pub.publish()

def handle_send(conn, data):
    while True:
        msg = input("[Server]: ")
        conn.sendall(msg.encode())

def main():
    rospy.init_node("tcp_jetracer_server")
    throttle_pub = rospy.Publisher(THROTTLE_TOPIC, Float32, queue_size=1)
    steering_pub = rospy.Publisher(STEERING_TOPIC, Float32, queue_size=1)
    pose_sub = rospy.Subscriber(POSE_TOPIC, PoseStamped, handle_send)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        print(f"Server listening on {HOST}:{PORT}...")
        conn, addr = s.accept()
        print(f"Connected by {addr}")

        threading.Thread(target=handle_receive, args=(conn,), daemon=True).start()
        handle_send(conn)

if __name__ == "__main__":
    main()