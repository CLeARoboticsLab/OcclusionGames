#!/usr/bin/env python3

import rospy
from std_msgs.msg import Float32
from geometry_msgs.msg import PoseStamped, TwistStamped
import csv
import sys

class DataCollector:
    def __init__(self):
        self.pose_sub = rospy.Subscriber('/vrpn_client_node/JaiAliJetRacer/pose', PoseStamped, pose_callback)
        self.twist_sub = rospy.Subscriber('/vrpn_client_node/JaiAliJetRacer/twist', TwistStamped, twist_callback)
        self.latest_pose = [0, 0, 0] # [x, y, psi]
        self.latest_twist = [0, 0, 0] # [vx, vy, psi_dot]
    def pose_callback(self, msg):
        _, _, psi = euler_from_quaternion(
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w
        )
        self.latest_pose = [msg.pose.position.x, msg.pose.position.y, psi]
    def twist_callback(self, msg):
        self.latest_twist = [msg.twist.linear.x, msg.twist.linear.y, msg.twist.angular.z]

def publish_from_csv(csv_path, rate_hz=100):
    # Initialize ROS node
    rospy.init_node('csv_to_ros_publisher', anonymous=True)

    # Define publishers
    throttle_pub = rospy.Publisher('/jetracer/throttle', Float32, queue_size=10)
    steering_pub = rospy.Publisher('/jetracer/steering', Float32, queue_size=10)

    # Define subscribers
    data_collector = DataCollector()

    # Prepare CSV logfile


    try:
        with open(csv_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            if 'u' not in reader.fieldnames or 'delta' not in reader.fieldnames:
                rospy.logerr("CSV must contain 'u' and 'delta' columns.")
                return

            rospy.loginfo("Publishing throttle (u) and steering (delta) data at {} Hz...".format(rate_hz))
            for row in reader:
                if rospy.is_shutdown():
                    break

                try:
                    throttle_val = float(row['u'])
                    steering_val = float(row['delta'])
                except ValueError:
                    rospy.logwarn("Invalid row skipped: %s", row)
                    continue

                throttle_pub.publish(Float32(throttle_val))
                steering_pub.publish(Float32(steering_val))
                rate.sleep()

            rospy.loginfo("Finished publishing all rows.")

    except IOError:
        rospy.logerr("Could not read file: %s", csv_path)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: csv_to_ros_publisher.py <path_to_csv>")
        sys.exit(1)
    csv_path = sys.argv[1]
    publish_from_csv(csv_path)
