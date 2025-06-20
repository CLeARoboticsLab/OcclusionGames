#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float32
from geometry_msgs.msg import PoseStamped
import math
import time
import csv

POSE_TOPIC = "vrpn_client_node/JaiAliJetRacer/pose"
ERR_EPSILON = 0.1
RADIUS = 0.75
CENTER = [-1.5, 1.5]
CHASING_DIST = 0.25
HEADING_BIAS = 0.04

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

        self.logfile = open("dynamics_csv/log_pidcircle.csv", "w")
        self.writer = csv.writer(self.logfile)
        self.writer.writerow(["Time", "x", "y","Error"])
        self.start_time = rospy.get_time()

        self.center = CENTER
        self.radius = RADIUS
        self.lookahead_distance = CHASING_DIST

        self.heading_pid = PIDController(0.15, 0, 0.02)
        self.distance_pid = PIDController(0.35, 0.0, 0.7)

    # def pose_callback(self, msg):
    #     x = msg.pose.position.x
    #     y = msg.pose.position.y
    #     q = msg.pose.orientation
    #     _, _, yaw = euler_from_quaternion(q.x, q.y, q.z, q.w)

    #     dx = self.goal[0] - x
    #     dy = self.goal[1] - y
    #     goal_dist = math.hypot(dx, dy)
    #     goal_heading = math.atan2(dy, dx)
    #     heading_error = self.angle_wrap(goal_heading - yaw)

    #     # Compute control
    #     steering = self.heading_pid.update(heading_error)
    #     throttle = self.distance_pid.update(goal_dist)

    #     # Limit values
    #     steering = max(min(steering, 1.0), -1.0)
    #     throttle = max(min(throttle, 0.2), 0.0) if goal_dist > ERR_EPSILON else 0.0

    #     self.steering_pub.publish(Float32(steering))
    #     self.throttle_pub.publish(Float32(throttle))

  # New callback
    def pose_callback(self, msg):
        x = msg.pose.position.x
        y = msg.pose.position.y
        q = msg.pose.orientation
        _, _, yaw = euler_from_quaternion(q.x, q.y, q.z, q.w)
    
        # Vector from center to car
        dx = x - self.center[0]
        dy = y - self.center[1]
        r = math.hypot(dx, dy)
        theta = math.atan2(dy, dx)

        ideal_point = [self.radius * math.cos(theta), self.radius * math.sin(theta)]

        now = rospy.get_time()

        position_error = math.hypot(x - ideal_point[0], y - ideal_point[1])
        self.writer.writerow([now - self.start_time, x, y, position_error])
        self.logfile.flush()

        radial_error = r - self.radius
    
        if abs(radial_error) > ERR_EPSILON:  # Phase 1: Converge to circle
            # Drive toward the closest point on the circle
            print("Need to go to circle")
            target_x = self.center[0] + self.radius * math.cos(theta)
            target_y = self.center[1] + self.radius * math.sin(theta)
        else:  # Phase 2: Follow the circle
            theta_target = theta - self.lookahead_distance / self.radius
            target_x = self.center[0] + self.radius * math.cos(theta_target)
            target_y = self.center[1] + self.radius * math.sin(theta_target)
    
        # Steer toward the target
        goal_heading = math.atan2(target_y - y, target_x - x)
        heading_error = self.angle_wrap(goal_heading - yaw)
    
        steering = self.heading_pid.update(heading_error)
        throttle = 0.15 if abs(radial_error) < 0.5 else 0.16  # Slightly faster until you get to the orbit
    
        steering = max(min(steering + HEADING_BIAS, 1.0), -1.0)
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
