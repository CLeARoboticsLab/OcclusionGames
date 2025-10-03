#!/usr/bin/env python3

import rospy
from std_msgs.msg import Float32
from geometry_msgs.msg import PoseStamped, TwistStamped
import csv
import sys
from DataCollector import DataCollector
import math

CSV_LOGFILE = "dynamics_csv/outside_collab_logfile.csv"

def publish_from_csv(csv_path, rate_hz=100):
    # Initialize ROS node
    rospy.init_node('csv_to_ros_publisher', anonymous=True)

    # Define publishers
    throttle_pub = rospy.Publisher('/jetracer/throttle', Float32, queue_size=10)
    steering_pub = rospy.Publisher('/jetracer/steering', Float32, queue_size=10)

    # Start data collector and wait for it to receive good pose data
    data_collector = DataCollector()
    while data_collector.latest_pose[0] == 0:
        pass # should not be perfectly 0

    # Prepare CSV logfile for data collection
    try:
        with open(CSV_LOGFILE, 'w') as logfile:
            writer = csv.writer(logfile)
            writer.writerow(['Time', 'steering', 'throttle', 'x', 'y', 'vx', 'vy', 'velocity', 'yaw', 'yaw_rate'])
            # Get initial time to compare to
            start_time = rospy.get_time()
            try:
                with open(csv_path, 'r') as csvfile:
                    reader = csv.DictReader(csvfile)
                    if 'u' not in reader.fieldnames or 'delta' not in reader.fieldnames:
                        print(reader.fieldnames)
                        rospy.logerr("CSV must contain 'u' and 'delta' columns.")
                        return
                    # Set rate
                    rate = rospy.Rate(rate_hz)
                    for row in reader:
                        if rospy.is_shutdown():
                            break
                        try:
                            throttle_val = float(row['u'])
                            steering_val = float(row['delta'])
                        except ValueError:
                            rospy.logwarn("Invalid row skipped: %s", row)
                            continue

                        # Publish throttle and steering values
                        throttle_pub.publish(Float32(throttle_val))
                        steering_pub.publish(Float32(steering_val))

                        # finally, write all data to the CSV log file
                        current_time = rospy.get_time() - start_time
                        pose = data_collector.latest_pose
                        twist = data_collector.latest_twist
                        velocity = math.hypot(twist[0], twist[1])
                        yaw = pose[2]
                        yaw_rate = twist[2]
                        writer.writerow([current_time, steering_val, throttle_val, pose[0], pose[1], twist[0], twist[1], velocity, yaw, yaw_rate])
                        rate.sleep()

                    rospy.loginfo("Finished publishing all rows.")
                    throttle_pub.publish(Float32(0.0))

            except IOError:
                rospy.logerr("Could not read file: %s", csv_path)
    except IOError:
        rospy.logerr("Could not open CSV log file for writing: %s", CSV_LOGFILE)
        return




if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: read_from_csv.py <path_to_csv>")
        sys.exit(1)
    csv_path = sys.argv[1]
    publish_from_csv(csv_path)
