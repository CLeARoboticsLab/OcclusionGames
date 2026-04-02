#!/usr/bin/env python3
"""Player-1 PID trajectory follower for JetRacer throttle/steering control.

This node tracks the first player's trajectory from OGSolve iLQG CSV output:
- Reads trajectory CSV (e.g., ilqg_trajectory_solution.csv) with header:
  time,p1_x,p1_y,p1_v,p1_theta,p1_a,p1_omega,p2_x,...
- Extracts player1 state (p1_x, p1_y, p1_v, p1_theta) only
- Uses feedforward actions (p1_a, p1_omega) are ignored; control is purely PID-based

It uses pose feedback (typically Vicon) and publishes:
- /player1/throttle_cmd (std_msgs/Float64)
- /player1/steering_cmd (std_msgs/Float64)
"""

import json
import math

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32


def wrap_angle(angle):
    """Wrap angle to [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))

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

class PID:
    def __init__(self, kp, ki, kd, i_limit=1.0):
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.i_limit = abs(float(i_limit))
        self.integral = 0.0
        self.prev_error = 0.0
        self.initialized = False

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.initialized = False

    def update(self, error, dt):
        if dt <= 0.0:
            return self.kp * error

        self.integral += error * dt
        self.integral = max(-self.i_limit, min(self.i_limit, self.integral))

        if not self.initialized:
            derivative = 0.0
            self.initialized = True
        else:
            derivative = (error - self.prev_error) / dt

        self.prev_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative


class Player1PidFollower:
    def __init__(self):
        rospy.init_node("player1_pid_follower", anonymous=False)

        self.trajectory_csv = rospy.get_param("~trajectory_csv", "~/catkin_ws/src/occlusion_games/trajectories/ilqg_trajectory_solution.csv")
        self.metadata_file = rospy.get_param("~metadata_file", "")

        self.pose_topic = rospy.get_param("~pose_topic", "/vrpn_client_node/og_jetracer1/pose")
        self.pose_msg_type = rospy.get_param("~pose_msg_type", "pose")  # pose | odom | transform

        self.throttle_topic = rospy.get_param("~throttle_topic", "/jetracer/throttle")
        self.steering_topic = rospy.get_param("~steering_topic", "/jetracer/steering")

        self.loop_rate_hz = float(rospy.get_param("~loop_rate", 100.0))
        self.default_dt = float(rospy.get_param("~trajectory_dt", 0.1))
        self.auto_start = bool(rospy.get_param("~auto_start", True))

        self.max_throttle = float(rospy.get_param("~max_throttle", 1.0))
        self.max_steering = float(rospy.get_param("~max_steering", 1.0))

        self.pid_speed = PID(
            kp=rospy.get_param("~pid_speed/kp", 0.05),
            ki=rospy.get_param("~pid_speed/ki", 0.01),
            kd=rospy.get_param("~pid_speed/kd", 0.2),
            i_limit=rospy.get_param("~pid_speed/i_limit", 0.25),
        )
        self.pid_lat = PID(
            kp=rospy.get_param("~pid_lat/kp", 0.25),
            ki=rospy.get_param("~pid_lat/ki", 0.0),
            kd=rospy.get_param("~pid_lat/kd", 0.02),
            i_limit=rospy.get_param("~pid_lat/i_limit", 1.0),
        )
        self.k_heading = float(rospy.get_param("~k_heading", 0.25))

        self.state_traj = self._load_state_csv(self.trajectory_csv)
        self.traj_dt = self._resolve_dt(self.default_dt, self.metadata_file)

        self.throttle_pub = rospy.Publisher(self.throttle_topic, Float32)
        self.steering_pub = rospy.Publisher(self.steering_topic, Float32)

        self.current_pose = None
        self.last_pose = None
        self.last_pose_time = None
        self.estimated_speed = 0.0

        if self.pose_msg_type == "pose":
            rospy.Subscriber(self.pose_topic, PoseStamped, self._pose_cb, queue_size=20)
        else:
            raise ValueError("pose_msg_type must be one of: pose, odom, transform")

        self.rate = rospy.Rate(self.loop_rate_hz)

        rospy.on_shutdown(self._stop_robot)
        rospy.loginfo("=" * 60)
        rospy.loginfo("Player1 PID follower initialized")
        rospy.loginfo("trajectory_csv=%s", self.trajectory_csv)
        rospy.loginfo("samples=%d dt=%.3f", len(self.state_traj), self.traj_dt)
        rospy.loginfo("pose_topic=%s (%s)", self.pose_topic, self.pose_msg_type)
        rospy.loginfo("throttle_topic=%s steering_topic=%s", self.throttle_topic, self.steering_topic)
        rospy.loginfo("=" * 60)

    def _load_state_csv(self, filename):
        """Load state trajectory from CSV with header parsing.
        
        CSV format (from iLQG solver):
            time,p1_x,p1_y,p1_v,p1_theta,p1_a,p1_omega,p2_x,p2_y,...
        
        Extracts only player1 state columns: [p1_x, p1_y, p1_v, p1_theta]
        Returns shape (n_steps, 4) with rows as [x, y, v, theta].
        """
        try:
            # Read with pandas to parse header easily
            import pandas as pd
            df = pd.read_csv(filename)
        except ImportError:
            rospy.logwarn("pandas not available; falling back to manual header parsing")
            # Manual parsing if pandas unavailable
            with open(filename, 'r') as f:
                header = f.readline().strip().split(',')
            data = np.loadtxt(filename, delimiter=",", skiprows=1)
        else:
            header = df.columns.tolist()
            data = df.values
        
        # Find indices for player1 state columns
        required_cols = ['p1_x', 'p1_y', 'p1_v', 'p1_theta']
        col_indices = []
        for col_name in required_cols:
            if col_name not in header:
                raise ValueError(f"Column '{col_name}' not found in CSV. Available: {header}")
            col_indices.append(header.index(col_name))
        
        # Extract only player1 state columns, skip time column
        state_data = data[:, col_indices]
        
        # Ensure 2D shape
        if state_data.ndim == 1:
            state_data = state_data.reshape(1, -1)
        
        rospy.loginfo("Loaded trajectory: %d steps, state shape %s", state_data.shape[0], state_data.shape)
        return state_data

    def _resolve_dt(self, default_dt, metadata_file):
        if not metadata_file:
            return default_dt
        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
            return float(meta.get("dt", default_dt))
        except Exception as exc:
            rospy.logwarn("Failed to read metadata file (%s): %s", metadata_file, exc)
            return default_dt

    def _extract_yaw(self, qx, qy, qz, qw):
        _, _, yaw = euler_from_quaternion(qx, qy, qz, qw)
        return float(yaw)

    def _update_pose(self, x, y, yaw):
        now = rospy.Time.now().to_sec()
        self.current_pose = (float(x), float(y), float(yaw), now)
        if self.last_pose is not None and self.last_pose_time is not None:
            dt = now - self.last_pose_time
            if dt > 1e-4:
                dx = self.current_pose[0] - self.last_pose[0]
                dy = self.current_pose[1] - self.last_pose[1]
                speed = math.hypot(dx, dy) / dt
                # Light smoothing to suppress Vicon jitter.
                self.estimated_speed = 0.9 * self.estimated_speed + 0.1 * speed
        self.last_pose = self.current_pose
        self.last_pose_time = now

    def _pose_cb(self, msg):
        p = msg.pose.position
        q = msg.pose.orientation
        yaw = self._extract_yaw(q.x, q.y, q.z, q.w)
        self._update_pose(p.x, p.y, yaw)

    def _stop_robot(self):
        self.throttle_pub.publish(Float32(data=0.0))
        self.steering_pub.publish(Float32(data=0.0))

    def run(self):
        if not self.auto_start:
            rospy.loginfo("auto_start is false. Set parameter to true to start execution.")
            return

        rospy.loginfo("Waiting for pose feedback...")
        while not rospy.is_shutdown() and self.current_pose is None:
            self.rate.sleep()

        if rospy.is_shutdown():
            return

        rospy.loginfo("Starting player1 trajectory tracking...")
        start_time = rospy.Time.now().to_sec()
        prev_loop_time = start_time

        n_states = self.state_traj.shape[0]

        while not rospy.is_shutdown():
            now = rospy.Time.now().to_sec()
            dt_loop = max(1e-4, now - prev_loop_time)
            prev_loop_time = now

            elapsed = now - start_time
            idx = int(elapsed / self.traj_dt)
            if idx >= n_states:
                rospy.loginfo("Trajectory complete. Sending zero command.")
                break

            x_ref = float(self.state_traj[idx, 0])
            y_ref = float(self.state_traj[idx, 1])
            v_ref = float(self.state_traj[idx, 2])
            yaw_ref = float(self.state_traj[idx, 3])

            x, y, yaw, _ = self.current_pose
            dx = x_ref - x
            dy = y_ref - y

            # Cross-track error in body coordinates.
            ey_body = -math.sin(yaw) * dx + math.cos(yaw) * dy
            yaw_err = wrap_angle(yaw_ref - yaw)

            speed_err = v_ref - self.estimated_speed
            throttle_cmd = self.pid_speed.update(speed_err, dt_loop)
            steering_lat = self.pid_lat.update(ey_body, dt_loop)
            steering_cmd = steering_lat + self.k_heading * yaw_err

            throttle_cmd = max(-self.max_throttle, min(self.max_throttle, throttle_cmd))
            steering_cmd = max(-self.max_steering, min(self.max_steering, steering_cmd))

            self.throttle_pub.publish(Float32(data=throttle_cmd))
            self.steering_pub.publish(Float32(data=steering_cmd))

            if idx % int(max(1.0, self.loop_rate_hz)) == 0:
                pos_err = math.hypot(dx, dy)
                rospy.loginfo(
                    "idx=%d pos_err=%.3f speed_err=%.3f yaw_err=%.3f thr=%.3f str=%.3f",
                    idx,
                    pos_err,
                    speed_err,
                    yaw_err,
                    throttle_cmd,
                    steering_cmd,
                )

            self.rate.sleep()

        self._stop_robot()


def main():
    try:
        node = Player1PidFollower()
        node.run()
    except rospy.ROSInterruptException:
        pass
    except Exception as exc:
        rospy.logerr("player1_pid_follower failed: %s", exc)
        raise


if __name__ == "__main__":
    main()
