#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float32
from geometry_msgs.msg import PoseStamped
import math
import time

POSE_TOPIC = "vrpn_client_node/JaiAliJetRacer/pose"
ERR_EPSILON = 0.1
GOAL = [-3.0, 3.0]

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
    def __init__(self):
        rospy.init_node('jetracer_pid_controller')
        self.steering_pub = rospy.Publisher('/jetracer/steering', Float32, queue_size=1)
        self.throttle_pub = rospy.Publisher('/jetracer/throttle', Float32, queue_size=1)
        rospy.Subscriber(POSE_TOPIC, PoseStamped, self.pose_callback)

        self.goal = GOAL  # example goal point in meters

        self.heading_pid = PIDController(1.0, 0.0, 0.65)
        self.distance_pid = PIDController(0.5, 0.0, 0.70) # Try pure proportinality first

    def pose_callback(self, msg):
        x = msg.pose.position.x
        y = msg.pose.position.y
        print('Postion: ', x, y)
        q = msg.pose.orientation
        _, _, yaw = euler_from_quaternion(q.x, q.y, q.z, q.w)

        dx = self.goal[0] - x
        dy = self.goal[1] - y
        goal_dist = math.hypot(dx, dy)
        goal_heading = math.atan2(dy, dx)
        heading_error = self.angle_wrap(goal_heading - yaw)

        # Compute control
        steering = self.heading_pid.update(heading_error)
        throttle = self.distance_pid.update(goal_dist)

        # Limit values
        steering = max(min(steering, 1.0), -1.0)
        throttle = max(min(throttle, 0.2), 0.0) if goal_dist > ERR_EPSILON else 0.0

        self.steering_pub.publish(Float32(steering))
        self.throttle_pub.publish(Float32(throttle))

    def angle_wrap(self, angle):
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

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


if __name__ == '__main__':
    controller = JetRacerController()
    rospy.spin()
