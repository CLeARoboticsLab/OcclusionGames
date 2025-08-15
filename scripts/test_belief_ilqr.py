"""
test_ilqr_unicycle.py
Smoke-test for the iLQR class with a 3-state unicycle model.

Requirements
------------
pip install "jax[cpu]" matplotlib
"""

from functools import partial
import jax
import jax.numpy as jnp
import math
import matplotlib.pyplot as plt
import time

# --------------------------------------------------------------------------- #
# ROS Setup for visualization
# --------------------------------------------------------------------------- #
import rospy
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Path
from std_msgs.msg import Header, ColorRGBA

rospy.init_node('trajectory_publisher', anonymous=True)
path_pub = rospy.Publisher('/planned_trajectory', Path, queue_size=1)
marker_pub = rospy.Publisher('/trajectory_markers', MarkerArray, queue_size=1)


# --------------------------------------------------------------------------- #
# 0.  Import your solver (change path if needed)
# --------------------------------------------------------------------------- #
import sys, os
sys.path.append(os.path.join(os.path.dirname(sys.path[0]), "belief_ilqr_solver"))
from solver import iLQR  # ← keep consistent with your repo

# imports for TCP
import socket
import json

HOST = "192.168.50.236"
PORT = 65432
RECV_CODE = "get_pose"


# --------------------------------------------------------------------------- #
# 1.  Dynamics & cost
# --------------------------------------------------------------------------- #
dt = 0.25  # [s]  integration step
T = 100  # horizon length
n, m = 4, 2  # state & control sizes

MASS = 2.5  # kg
LENGTH = 0.26 # wheelbase
B_U_1 = 8.0  # m/s² (≈0.2 g per unit throttle)
B_U_2 = 4.5  # m/s² (≈0.2 g per unit throttle)
B_DELTA = 0.46
THROTTLE_MAX = 0.25 # max throttle without slipping
HEADING_BIAS_1 = -0.2  # steering bias
U_L = 1.25
U_A = 0.25
U_B = 15


@jax.jit
def dynamics(state: jnp.ndarray, control: jnp.ndarray) -> jnp.ndarray:
    """
    Bicycle discrete-time dynamics for JetRacer
    state   = [x, y, v, θ]
    control = [u, δ]      <-- throttle and steering commands
    """
    # unpack state, control
    x, y, vx, th = state
    u_raw, delta_raw = control
    k = 100
    # clip controls
    u_raw = U_L / (1 + U_A * jnp.exp(-U_B * u_raw)) - 1  # squashes throttle via sigmoid
    u_raw = jnp.log(1 + jnp.exp(k * u_raw)) / k  # squashes throttle via log
    delta_raw += HEADING_BIAS_1
    delta_raw = jnp.tanh(delta_raw)  # clamp steering to [-1, 1] via tanh

    # compute state changes
    x_dot = vx * jnp.cos(th)
    y_dot = vx * jnp.sin(th)
    vx_dot = B_U_1 * u_raw
    th_dot = vx / LENGTH * jnp.tan(B_DELTA * delta_raw)

    # update state
    x_next = x + x_dot * dt
    y_next = y + y_dot * dt
    v_next = vx + vx_dot * dt
    th_next = th + th_dot * dt

    return jnp.array([x_next, y_next, v_next, th_next])


# -- single-step quadratic-ish cost -------------------------------------------------
w_pos, w_th, w_u = 1.0, 1.0, 10.0


# def stage_cost(x, u):
#     pos_cost = w_pos * (x[0] ** 2 + x[1] ** 2)
#     head_cost = w_th * (x[2] ** 2)
#     ctrl_cost = w_u * jnp.sum(u**2)
#     # Penalize negative velocity command (u[0])
#     reverse_penalty = 100.0 * jnp.maximum(-u[0], 0.0)  # Large penalty for negative throttle
#     return pos_cost + head_cost + ctrl_cost + reverse_penalty


# -- wrapper that works for both: single step  *and*  full trajectory --------------
# def build_unicycle_cost(w_pos=1.0, w_th=0.1, w_u=1e-2, term_weight=100.0):
#     """
#     Returns a dict with 'stage', 'terminal', 'traj' functions
#     compatible with the revised iLQR constructor.
#     """

#     # ---------------- stage cost  ℓ(x,u) ------------------------------------
#     @jax.jit
#     def stage_cost(x, u):
#         pos = w_pos * (x[0] ** 2 + x[1] ** 2)
#         heading = w_th * (x[2] ** 2)
#         control = w_u * jnp.sum(u**2)
#         # Penalize negative velocity command (u[0])
#         reverse_penalty = 10000.0 * jnp.maximum(-u[0], 0.0)  # Large penalty for negative throttle
#         return pos + heading + control + reverse_penalty

#     # -------------- terminal cost Φ(x_T) ------------------------------------
#     @jax.jit
#     def terminal_cost(x_T):
#         return term_weight * (x_T[0] ** 2 + x_T[1] ** 2 + 0.1 * x_T[2] ** 2)

#     # -------------- convenience: full trajectory cost -----------------------
#     @jax.jit
#     def traj_cost(xs, us):
#         step_costs = jax.vmap(stage_cost)(xs[:-1], us)  # (T,)
#         return jnp.sum(step_costs) + terminal_cost(xs[-1])

#     return {"stage": stage_cost, "terminal": terminal_cost, "traj": traj_cost}

# -- build circle cost function ----------------------------------------------
def build_circle_cost(
    center=(0.0, 0.0),
    radius=1.0,
    v_ref=0.5,
    theta0=0.0,
    dt=0.02,
    w_pos=1.0,
    w_th=0.1,
    w_v=0.0,      # weight for velocity error (optional)
    w_u=1e-2,
    term_weight=100.0
):
    """
    Returns {'stage', 'terminal', 'traj'} cost functions for iLQR
    tracking a circular trajectory.
    State: x = [x, y, v, yaw]
    Control: u = [v_cmd, omega]
    """

    center = jnp.array(center)
    omega_ref = v_ref / radius
    t_counter = {"t": 0}

    def wrap_angle(a):
        return (a + jnp.pi) % (2.0 * jnp.pi) - jnp.pi

    def ref_at_t(t):
        ang = theta0 + omega_ref * (t * dt)
        x_ref = jnp.array([
            center[0] + radius * jnp.cos(ang),   # x
            center[1] + radius * jnp.sin(ang),   # y
            v_ref,                               # v
            ang + jnp.pi / 2.0                   # yaw
        ])
        u_ref = jnp.array([v_ref, omega_ref])
        return x_ref, u_ref

    def stage_cost(x, u):
        t = t_counter["t"]
        x_ref, u_ref = ref_at_t(t)

        # position error
        pos_err = x[0] - x_ref[0], x[1] - x_ref[1]
        pos_cost = w_pos * (pos_err[0]**2 + pos_err[1]**2)

        # heading error
        yaw_err = wrap_angle(x[3] - x_ref[3])
        head_cost = w_th * (yaw_err**2)

        # velocity error (optional, if w_v > 0)
        vel_err = x[2] - x_ref[2]
        vel_cost = w_v * (vel_err**2)

        # control cost
        ctrl_cost = w_u * jnp.sum((u - u_ref) ** 2)

        t_counter["t"] += 1
        return pos_cost + head_cost + vel_cost + ctrl_cost

    def terminal_cost(x_T):
        t_T = t_counter["t"]
        x_ref, _ = ref_at_t(t_T)
        pos_err = x_T[0] - x_ref[0], x_T[1] - x_ref[1]
        yaw_err = wrap_angle(x_T[3] - x_ref[3])
        vel_err = x_T[2] - x_ref[2]
        return term_weight * (
            pos_err[0]**2 + pos_err[1]**2 + 0.1 * yaw_err**2 + w_v * vel_err**2
        )

    def traj_cost(xs, us):
        t_counter["t"] = 0
        step_costs = jax.vmap(stage_cost)(xs[:-1], us)
        return jnp.sum(step_costs) + terminal_cost(xs[-1])

    return {"stage": stage_cost, "terminal": terminal_cost, "traj": traj_cost}


cost = build_circle_cost(w_pos=w_pos, w_th=w_th, w_u=w_u, term_weight=300.0)

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
    # --------------------------------------------------------------------------- #
    # 2.  Problem setup
    # --------------------------------------------------------------------------- #
    # get pose information from the server
    client_socket.connect((HOST, PORT)) 
    client_socket.sendall(RECV_CODE.encode("utf-8"))
    data = client_socket.recv(1024).decode("utf-8")
    pose_data = json.loads(data)
    x0 = jnp.array([pose_data["x"], pose_data["y"], 0, pose_data["psi"]])  # initial state
    u_init = jnp.zeros((T, m))  # zero-controls guess
    print(f"Initial state: {x0}")

    # Instantiate solver
    dims = {"state": n, "control": m}
    ilqr = iLQR(cost, dynamics, T, dims)

    # --------------------------------------------------------------------------- #
    # 3.  Solve
    # --------------------------------------------------------------------------- #
    start_time = time.time()
    (states, controls), (success, stats) = ilqr.solve(x0, u_init)

    ilqr.print_stats(stats.block_until_ready())
    print(f"iLQR solve time: {time.time() - start_time:.2f} s")

    states_np = jax.device_get(states)
    controls_np = jax.device_get(controls)

    print(f"Success flag   : {success}")
    print(f"Final state    : {states_np[-1]}")
    print(f"Distance to 0  : {jnp.linalg.norm(states_np[-1][:2]):.4f}  m")

    # --------------------------------------------------------------------------- #
    # 4.  Send/execute controls over TCP
    # --------------------------------------------------------------------------- #
    
    # Low-pass filter parameters
    alpha = 0.3  # filter coefficient (0 < alpha < 1, smaller = more filtering)
    filtered_throttle = 0.0  # initialize filtered values
    filtered_steering = 0.0
    
    for i, (state, control) in enumerate(zip(states_np, controls_np)): # iterate over all controls
        curr_throttle = float(control[0])
        curr_steering = float(control[1])
        
        # Apply low-pass filter (exponential moving average)
        if i == 0:
            # First iteration: initialize with current values
            filtered_throttle = curr_throttle
            filtered_steering = curr_steering
        else:
            # Apply filter: filtered = alpha * current + (1-alpha) * previous_filtered
            filtered_throttle = alpha * curr_throttle + (1 - alpha) * filtered_throttle
            filtered_steering = alpha * curr_steering + (1 - alpha) * filtered_steering
        
        # clip controls if not already
        filtered_throttle = min(THROTTLE_MAX, max(0, filtered_throttle))
        filtered_steering = min(1, max(-1, filtered_steering))
        
        # send data over the network
        control_dict = {"throttle": filtered_throttle, "steering": filtered_steering}
        print("State:", state)
        print("Unclipped control:", control)
        print("Filtered control:", control_dict)
        print()
        json_str = json.dumps(control_dict)
        client_socket.sendall(json_str.encode("utf-8"))
        # now, wait for dt number of ms to send the next command
        time.sleep(dt)


# --------------------------------------------------------------------------- #
# 5.  Publish trajectory to RViz
# --------------------------------------------------------------------------- #

def publish_trajectory_to_rviz(states_np, frame_id="map"):
    """
    Publish the planned trajectory to RViz as both a Path and MarkerArray
    """
    # Create Path message
    path_msg = Path()
    path_msg.header = Header()
    path_msg.header.stamp = rospy.Time.now()
    path_msg.header.frame_id = frame_id
    
    # Create MarkerArray for better visualization
    marker_array = MarkerArray()
    
    # Line strip marker for the trajectory
    line_marker = Marker()
    line_marker.header = path_msg.header
    line_marker.ns = "trajectory_line"
    line_marker.id = 0
    line_marker.type = Marker.LINE_STRIP
    line_marker.action = Marker.ADD
    line_marker.scale.x = 0.05  # line width
    line_marker.color = ColorRGBA(1.0, 0.0, 0.0, 1.0)  # red
    
    # Arrow markers for heading
    for i, state in enumerate(states_np):
        x, y, v, theta = state
        
        # Add to path
        pose = PoseStamped()
        pose.header = path_msg.header
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.position.z = 0.0
        
        # Convert heading to quaternion
        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 0.0
        pose.pose.orientation.z = jnp.sin(theta / 2.0)
        pose.pose.orientation.w = jnp.cos(theta / 2.0)
        
        path_msg.poses.append(pose)
        
        # Add point to line marker
        point = Point()
        point.x = float(x)
        point.y = float(y)
        point.z = 0.0
        line_marker.points.append(point)
        
        # Add arrow markers every few points for heading visualization
        if i % 3 == 0:  # every 3rd point
            arrow_marker = Marker()
            arrow_marker.header = path_msg.header
            arrow_marker.ns = "trajectory_arrows"
            arrow_marker.id = i
            arrow_marker.type = Marker.ARROW
            arrow_marker.action = Marker.ADD
            arrow_marker.pose = pose.pose
            arrow_marker.scale.x = 0.2  # length
            arrow_marker.scale.y = 0.05  # width
            arrow_marker.scale.z = 0.05  # height
            arrow_marker.color = ColorRGBA(0.0, 1.0, 0.0, 0.8)  # green
            marker_array.markers.append(arrow_marker)
    
    marker_array.markers.append(line_marker)
    
    # Publish messages
    path_pub.publish(path_msg)
    marker_pub.publish(marker_array)
    print("Published trajectory to RViz")

publish_trajectory_to_rviz(states_np, frame_id="map")

# --------------------------------------------------------------------------- #
# 6.  Plots
# --------------------------------------------------------------------------- #
t = jnp.arange(T + 1) * dt

fig, axs = plt.subplots(3, 1, figsize=(8, 8))

# (a) XY trajectory
axs[0].plot(states_np[:, 0], states_np[:, 1], marker="o", ms=2)
axs[0].set_title("Unicycle XY path")
axs[0].set_xlabel("x [m]")
axs[0].set_ylabel("y [m]")
axs[0].axis("equal")
axs[0].grid(alpha=0.3)

# (b) heading + positions vs time
axs[1].plot(t, states_np[:, 0], label="x")
axs[1].plot(t, states_np[:, 1], label="y")
axs[1].plot(t, states_np[:, 2], label="θ", linestyle="--")
axs[1].set_title("State trajectories"), axs[1].grid(alpha=0.3)
axs[1].legend(ncol=3)

# (c) controls
axs[2].step(t[:-1], controls_np[:, 0], where="post", label="v")
axs[2].step(t[:-1], controls_np[:, 1], where="post", label="ω")
axs[2].set_title("Control inputs")
axs[2].set_xlabel("time [s]")
axs[2].set_ylabel("command")
axs[2].grid(alpha=0.3)
axs[2].legend()

plt.tight_layout()
plt.savefig("figures/ex_5_ilqr.png", dpi=300)
plt.close()
