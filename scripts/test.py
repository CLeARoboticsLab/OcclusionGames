#!/usr/bin/env python3
# This whole shebang nonsense is annoying
import rospy
from std_msgs.msg import Float32
import numpy as np
import time

def stuff():
    rospy.init_node("Data_Collecttor")

    steering_pub = rospy.Publisher("/jetracer/steering", Float32, queue_size=1)
    throttle_pub = rospy.Publisher("/jetracer/throttle", Float32, queue_size=1)

    rospy.sleep(1.0)

    throttle_vals = np.linspace(0.05, 0.25, 15)
    steering_vals = np.linspace(0.05, 1.0, 10)

    duration = 2.0
    rest_time = 1.0

    for throttle in throttle_vals:
        for steering in steering_vals:
            rospy.loginfo(f"Throttle: {throttle:.2f}, Steering: {steering:.2f}")
            throttle_pub.publish(Float32(throttle))
            steering_pub.publish(Float32(steering))
            rospy.sleep(duration)

            # Reset the hardware. May make the measurement somewhat noisy
            throttle_pub.publish(Float32(0.0))
            steering_pub.publish(Float32(0.0))
            rospy.sleep(rest_time)

    # Done
    throttle_pub.publsih(Float32(0.0))
    steering_pub.publish(Float32(0.0))
    rospy.loginfo("Data Sweep Complete")

if __name__ == '__main__':
    try:
        stuff()
    except:
        pass
