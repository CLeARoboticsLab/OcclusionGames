#!/usr/bin/env python3

import rospy
from std_msgs.msg import Float32
from geometry_msgs.msg import PoseStamped
import math
import time
import csv
from aura_circle import PIDController, JetRacerController, euler_from_quaternion
from vicon_log import ViconLogger

if __name__ == '__main__':
    logger = ViconLogger()
    controller = JetRacerController()
    rospy.spin()
