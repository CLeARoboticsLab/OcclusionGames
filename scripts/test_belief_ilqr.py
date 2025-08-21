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
import numpy as np
import math
import matplotlib.pyplot as plt
import time

# --------------------------------------------------------------------------- #
# ROS Setup for visualization
# --------------------------------------------------------------------------- #
#import rospy
#from geometry_msgs.msg import PoseStamped, Point
#from visualization_msgs.msg import Marker, MarkerArray
#from nav_msgs.msg import Path
#from std_msgs.msg import Header, ColorRGBA

#rospy.init_node('trajectory_publisher', anonymous=True)
#path_pub = rospy.Publisher('/planned_trajectory', Path, queue_size=1)
#marker_pub = rospy.Publisher('/trajectory_markers', MarkerArray, queue_size=1)


# --------------------------------------------------------------------------- #
# 0.  Import your solver (change path if needed)
# --------------------------------------------------------------------------- #
import sys, os
sys.path.append(os.path.join(os.path.dirname(sys.path[0]), "belief_ilqr_solver"))
from solver import iLQR  # ← keep consistent with your repo
from solver_with_time import iLQR_t  # ← keep consistent with your repo

# imports for TCP
import socket
import json

DEBUGGING_MODE = True  # if True, use fixed initial state and simulate; otherwise, get from server and send controls over TCP
TIME_REQ = True # if True, use iLQR_t; otherwise, use iLQR
USING_RK4_DYNAMICS = True # if True, use Runge-Kutta 4 to update state dynamics
USING_RK4_GRAPHING = False # if True, use Runge-Kutta 4 to graph state trajectory
USING_FULL_DYNAMICS = False # if True, use full dynamics; otherwise, use simplified bicycle model

HOST = "192.168.50.236"
PORT = 65432
RECV_CODE = "get_pose"


# --------------------------------------------------------------------------- #
# 1.  Dynamics & cost
# --------------------------------------------------------------------------- #
dt = 0.075  # [s]  integration step
T = 60  # horizon length
V_REF = 2.5 # [m/s]  reference velocity for the circle cost
VX_INIT = 0.01 #[m/s]
CIRCLE_RADIUS = 2.0 # [m]  radius of the circle to track
n = 6 if USING_FULL_DYNAMICS else 4 # state size
m = 2  # control size (throttle, steering)

MASS = 2.5  # kg
LENGTH = 0.26 # wheelbase
GRAVITY = 9.81  # m/s²
ROLLING_FRICTION_COEFF = 0.0 # estimate for rolling friction
COM_TO_FRONT = 0.082 # distance from center of mass to front axle (m)
COM_TO_REAR = LENGTH - COM_TO_FRONT
YAW_INERTIA = 0.015 # kg*m²
C_ALPHA_F = 2.0 # cornering stiffness front (N/rad)
C_ALPHA_R = 2.0 # cornering stiffness rear (N/rad)
B_U_1 = 5.0  # m/s² (≈0.2 g per unit throttle)
B_U_2 = 4.5  # m/s² (≈0.2 g per unit throttle)
B_DELTA = 0.4
THROTTLE_MAX = 0.5 # max throttle without slipping
HEADING_BIAS_1 = -0.0  # steering bias
U_L = 1.25
U_A = 0.25
U_B = 15

# -- Runge-Kutta 4 Solver ------------------------------------------
def runge_kutta_4(f, y0: np.ndarray, t0, tf, h):
    """
    4th-order Runge-Kutta method for numerical integration.
    Parameters:
        f: first-order derivative of y, written as f(t, y)
        y0: initial value of the function to approximate
        t0: initial time
        tf: final time
        h: step size
    Returns:
        (t_values, y_values): array of timesteps and their respective approximated function values
    """
    # Create an array for time steps
    t_values = np.arange(t0, tf + h, h)
    y_values = np.zeros((len(t_values), len(y0)))
    # Set the initial condition
    y_values[0] = y0
    # Perform the RK4 iteration
    for i in range(1, len(t_values)):
        t = t_values[i - 1]
        y = y_values[i - 1]
        k1 = h * f(t, y)
        k2 = h * f(t + h / 2, y + k1 / 2)
        k3 = h * f(t + h / 2, y + k2 / 2)
        k4 = h * f(t + h, y + k3)
        #print ("K values: ", k1, k2, k3, k4)
        # Update y based on the RK4 formula
        y_values[i] = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return t_values, y_values

@jax.jit
def dynamics(state: jnp.ndarray, control: jnp.ndarray) -> jnp.ndarray:
    """
    Bicycle discrete-time dynamics for JetRacer
    state   = [x, y, v, θ]
    control = [u, δ]      <-- throttle and steering commands
    """
    def drag_force(v,
                Crr=0.015, m=MASS, g=GRAVITY,
                Fc=0.05, Fs=0.07, vs=0.08,
                c1=0.08, c2=0.02,
                eps=1e-3, delta=1e-3):
        """
        Returns longitudinal drag/friction force [N] opposing velocity v [m/s].
        Smooth & branchless; suitable for differentiable simulators.
        """
        sgn_smooth = jnp.tanh(v/eps)                  # smooth sign
        abs_smooth = jnp.sqrt(v*v + delta*delta)      # smooth |v|
        stribeck = Fc + (Fs - Fc) * jnp.exp(-(abs_smooth/vs)**2)
        rolling_plus_contact = Crr*m*g + stribeck
        final_force = - sgn_smooth * rolling_plus_contact - c1*v - c2*v*abs_smooth
        #jax.debug.print("Drag force: {final_force}", final_force=final_force)
        return final_force

    def d_state(curr_state: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the current state derivative.
        Returns:
            state_dot: state derivative vector [x_dot, y_dot, v_dot, θ_dot]
        """
        x, y, th, vx, vy, om = None, None, None, None, None, None
        if USING_FULL_DYNAMICS:
            x, y, th, vx, vy, om = curr_state
        else:
            x, y, vx, th = curr_state
        u_raw, delta_raw = control
        k = 100
        # clip controls
        #u_raw = jnp.tanh(u_raw)
        #u_raw = jnp.maximum(u_raw, 0.0) #ensure throttle is not too low
        #u_raw = U_L / (1 + U_A * jnp.exp(-U_B * u_raw)) - 1  # squashes throttle via sigmoid
        #u_raw = jnp.log(1 + jnp.exp(k * (u_raw))) / k - 0.007  # squashes throttle via log
        #delta_raw += HEADING_BIAS_1
        #delta_raw = jnp.tanh(delta_raw)  # clamp steering to [-1, 1] via tanh

        state_dot = None
        if USING_FULL_DYNAMICS:
            ### NEW DYNAMICS ###
            # position and orientation
            c, s = jnp.cos(th), jnp.sin(th)
            x_dot = vx * c - vy * s
            y_dot = vx * s + vy * c
            th_dot = om

            # velocity in body frame and yaw rate
            slip_f = B_DELTA * delta_raw - (vy + COM_TO_FRONT * om) / vx
            slip_r = - (vy + COM_TO_REAR * om) / vx
            F_x = MASS * B_U_1 * u_raw
            F_y_f = -C_ALPHA_F * slip_f
            F_y_r = -C_ALPHA_R * slip_r
            vx_dot = (F_x / MASS) - om * vy
            vy_dot = (F_y_f + F_y_r) / MASS + om * vx
            om_dot = (COM_TO_FRONT * F_y_f - COM_TO_REAR * F_y_r) / YAW_INERTIA
            state_dot = jnp.array([x_dot, y_dot, th_dot, vx_dot, vy_dot, om_dot])
        else:
            ### OLD DYNAMICS ###
            # compute state changes
            x_dot = vx * jnp.cos(th)
            y_dot = vx * jnp.sin(th)
            #v_dot = B_U_1 * u_raw - drag_force(v) / MASS  # account for rolling friction
            vx_dot = B_U_1 * u_raw - ROLLING_FRICTION_COEFF * vx * vx / LENGTH  # account for rolling friction
            th_dot = vx / LENGTH * jnp.tan(B_DELTA * delta_raw)
            state_dot = jnp.array([x_dot, y_dot, vx_dot, th_dot])
        return state_dot
    next_state = None
    if USING_RK4_DYNAMICS:
        # apply RK4
        k1 = dt * d_state(state)
        k2 = dt * d_state(state + k1 / 2)
        k3 = dt * d_state(state + k2 / 2)
        k4 = dt * d_state(state + k3)
        # Update y based on the RK4 formula
        next_state = state + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    else:
        # apply Euler integration
        next_state = state + dt * d_state(state)

    return next_state


# -- single-step quadratic-ish cost -------------------------------------------------


# def stage_cost(x, u):
#     pos_cost = w_pos * (x[0] ** 2 + x[1] ** 2)
#     head_cost = w_th * (x[2] ** 2)
#     ctrl_cost = w_u * jnp.sum(u**2)
#     # Penalize negative velocity command (u[0])
#     reverse_penalty = 100.0 * jnp.maximum(-u[0], 0.0)  # Large penalty for negative throttle
#     return pos_cost + head_cost + ctrl_cost + reverse_penalty


# -- wrapper that works for both: single step  *and*  full trajectory --------------
def build_unicycle_cost(w_pos=1.0, w_th=0.1, w_u=1e-2, term_weight=100.0):
    """
    Returns a dict with 'stage', 'terminal', 'traj' functions
    compatible with the revised iLQR constructor.
    """

    # ---------------- stage cost  ℓ(x,u) ------------------------------------
    @jax.jit
    def stage_cost(x, u):
        pos = w_pos * (x[0] ** 2 + x[1] ** 2)
        heading = w_th * (x[2] ** 2)
        control = w_u * jnp.sum(u**2)
        # Penalize negative velocity command (u[0])
        #reverse_penalty = 10000.0 * jnp.maximum(-u[0], 0.0)  # Large penalty for negative throttle
        return pos + heading + control #+ reverse_penalty

    # -------------- terminal cost Φ(x_T) ------------------------------------
    @jax.jit
    def terminal_cost(x_T):
        return term_weight * (x_T[0] ** 2 + x_T[1] ** 2 + 0.1 * x_T[2] ** 2)

    # -------------- convenience: full trajectory cost -----------------------
    @jax.jit
    def traj_cost(xs, us):
        step_costs = jax.vmap(stage_cost)(xs[:-1], us)  # (T,)
        return jnp.sum(step_costs) + terminal_cost(xs[-1])

    return {"stage": stage_cost, "terminal": terminal_cost, "traj": traj_cost}

# -- build circle cost function ----------------------------------------------
def build_circle_cost(
    center=(0.0, 0.0),
    radius=1.0,
    v_ref=0.5,
    theta0=0.0,
    dt=0.02,
    w_pos=500.0,     # Increase position weight
    w_th=10.0,      # Increase heading weight
    w_v=5.0,        # Adjust velocity weight
    w_u=1.0,        # Increase control penalty
    w_omega=2.0,    # Reduce angular velocity weight
    reverse_penalty=0.0,
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

    def wrap_angle(a):
        return (a + jnp.pi) % (2.0 * jnp.pi) - jnp.pi
        

    def ref_at_t(t):
        ang = theta0 + omega_ref * (t * dt)
        N = 10  # number of steps to ramp up
        v_ref_t = v_ref * jnp.clip(t / N, 0, 1)
        omega_ref_t = v_ref_t / radius
        x_ref = jnp.array([
            center[0] + radius * jnp.cos(ang),
            center[1] + radius * jnp.sin(ang),
            v_ref_t,
            ang + jnp.pi / 2.0
        ])
        u_ref = jnp.array([v_ref_t, omega_ref_t])
        return x_ref, u_ref

    @jax.jit
    def stage_cost(x, u, t):
        x_ref, u_ref = ref_at_t(t)
        
        # Position tracking
        pos_err = jnp.array([x[0] - x_ref[0], x[1] - x_ref[1]])
        pos_cost = w_pos * jnp.sum(pos_err**2)
        
        # Heading tracking (with proper angle wrapping)
        yaw_err = wrap_angle(x[3] - x_ref[3])
        head_cost = w_th * yaw_err**2
        
        # Velocity tracking
        vel_cost = w_v * ((x[2] - x_ref[2])**2)
        
        # Control effort (relative to reference)
        ctrl_cost = w_u * jnp.sum(u**2)
        
        # Reverse penalty
        rev_penalty = reverse_penalty * jnp.maximum(-u[0], 0.0)
        
        # Angular velocity tracking
        omega_cost = w_omega * ((u[1] - u_ref[1])**2)

        return pos_cost + head_cost + vel_cost + ctrl_cost + rev_penalty + omega_cost

    @jax.jit
    def terminal_cost(x_T, t_T):
        x_ref, _ = ref_at_t(t_T)
        pos_cost = w_pos * ((x_T[0] - x_ref[0])**2 + (x_T[1] - x_ref[1])**2)
        yaw_err = wrap_angle(x_T[3] - x_ref[3])
        head_cost = w_th * (yaw_err**2)
        vel_cost = w_v * ((x_T[2] - x_ref[2])**2)
        return term_weight * (pos_cost + 0.1 * head_cost + vel_cost)

    @jax.jit
    def traj_cost(xs, us):
        T = us.shape[0]
        t_arr = jnp.arange(T)
        step_costs = jax.vmap(stage_cost)(xs[:-1], us, t_arr)
        return jnp.sum(step_costs) + terminal_cost(xs[-1], T)

    return {
        "stage": lambda x, u, t: stage_cost(x, u, t),
        "terminal": lambda x_T: terminal_cost(x_T, T),
        "traj": traj_cost
    }

def build_wavepoint_cost(
    center=(0.0, 0.0),
    radius=1.0,
    v_ref=0.5,
    theta0=0.0,
    dt=0.02,
    w_pos=500.0,     # Increase position weight
    w_th=10.0,      # Increase heading weight
    w_v=5.0,        # Adjust velocity weight
    w_u=1.0,        # Increase control penalty
    w_omega=2.0,    # Reduce angular velocity weight
    reverse_penalty=0.0,
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

    def wrap_angle(a):
        return (a + jnp.pi) % (2.0 * jnp.pi) - jnp.pi
        

    def ref_at_t(t):
        ang = theta0 + omega_ref * (t * dt)
        N = 10  # number of steps to ramp up
        v_ref_t = v_ref # * jnp.clip(t / N, 0, 1)
        omega_ref_t = v_ref_t / radius
        x_ref = jnp.array([
            center[0] + radius * jnp.cos(ang),
            center[1] + radius * jnp.sin(ang),
            v_ref_t,
            ang + jnp.pi / 2.0
        ])
        u_ref = jnp.array([v_ref_t, omega_ref_t])
        return x_ref, u_ref

    @jax.jit
    def stage_cost(x, u, t):
        x_ref, u_ref = ref_at_t(t)
        
        # Position tracking
        pos_err = jnp.array([x[0] - x_ref[0], x[1] - x_ref[1]])
        pos_cost = w_pos * jnp.sum(pos_err**2)
        
        # Heading tracking (with proper angle wrapping)
        yaw_err = wrap_angle(x[3] - x_ref[3])
        head_cost = w_th * yaw_err**2
        
        # Velocity tracking
        vel_cost = w_v * ((x[2] - x_ref[2])**2)
        
        # Control effort (relative to reference)
        ctrl_cost = w_u * jnp.sum((u - u_ref)**2)
        
        # Reverse penalty
        rev_penalty = reverse_penalty * jnp.maximum(-u[0], 0.0)
        
        # Angular velocity tracking
        omega_cost = w_omega * ((u[1] - u_ref[1])**2)

        return pos_cost + head_cost + vel_cost + ctrl_cost + rev_penalty + omega_cost

    @jax.jit
    def terminal_cost(x_T, t_T):
        x_ref, _ = ref_at_t(t_T)
        pos_cost = w_pos * ((x_T[0] - x_ref[0])**2 + (x_T[1] - x_ref[1])**2)
        yaw_err = wrap_angle(x_T[3] - x_ref[3])
        head_cost = w_th * (yaw_err**2)
        vel_cost = w_v * ((x_T[2] - x_ref[2])**2)
        return term_weight * (pos_cost + 0.1 * head_cost + vel_cost)

    @jax.jit
    def traj_cost(xs, us):
        T = us.shape[0]
        t_arr = jnp.arange(T)
        step_costs = jax.vmap(stage_cost)(xs[:-1], us, t_arr)
        return jnp.sum(step_costs) + terminal_cost(xs[-1], T)

    return {
        "stage": lambda x, u, t: stage_cost(x, u, t),
        "terminal": lambda x_T: terminal_cost(x_T, T),
        "traj": traj_cost
    }


reg_cost = build_unicycle_cost(
    w_pos=500.0,     # Stronger position tracking
    w_th=10.0,      # Moderate heading tracking
    w_u=10.0,        # Moderate control penalty
    term_weight=100.0  # Strong terminal cost
)

circle_cost = build_circle_cost(
    radius=CIRCLE_RADIUS,
    v_ref=V_REF,      # Slower reference velocity
    dt=dt,
    w_pos=10.0,     # Stronger position tracking
    w_th=5.0,      # Moderate heading tracking
    w_v=50.0,        # Moderate velocity tracking
    w_u=25.0,        # Moderate control penalty
    w_omega=40.0,     # Lower angular velocity weight
    term_weight=100.0
)

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
    # --------------------------------------------------------------------------- #
    # 2.  Problem setup
    # --------------------------------------------------------------------------- #
    # get pose information from the server
    x0 = None
    if DEBUGGING_MODE:
        if USING_FULL_DYNAMICS:
            x0 = jnp.array([CIRCLE_RADIUS, 0, jnp.pi / 2, VX_INIT, 0.00, 0.0])  # fixed initial state
        else:
            x0 = jnp.array([CIRCLE_RADIUS, 0, VX_INIT, jnp.pi / 2])  # fixed initial state
    else:
        client_socket.connect((HOST, PORT)) 
        client_socket.sendall(RECV_CODE.encode("utf-8"))
        data = client_socket.recv(1024).decode("utf-8")
        pose_data = json.loads(data)
        if USING_FULL_DYNAMICS:
            x0 = jnp.array([pose_data["x"], pose_data["y"], pose_data["psi"], VX_INIT, 0.0, 0.0])  # initial state
        else:
            x0 = jnp.array([pose_data["x"], pose_data["y"], VX_INIT, pose_data["psi"]])  # initial state
    u_init = jnp.zeros((T, m))  # zero-controls guess
    print(f"Initial state: {x0}")

    # Instantiate solver
    dims = {"state": n, "control": m}
    ilqr = iLQR(reg_cost, dynamics, T, dims)
    ilqr_t = iLQR_t(circle_cost, dynamics, T, dims)
    ilqr_to_use = ilqr_t if TIME_REQ else ilqr

    # --------------------------------------------------------------------------- #
    # 3.  Solve
    # --------------------------------------------------------------------------- #
    start_time = time.time()
    (states, controls), (success, stats) = ilqr_to_use.solve(x0, u_init)

    ilqr_to_use.print_stats(stats.block_until_ready())
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
        if not DEBUGGING_MODE:
            client_socket.sendall(json_str.encode("utf-8"))
            # now, wait for dt number of ms to send the next command
            time.sleep(dt)


# --------------------------------------------------------------------------- #
# 5.  Publish trajectory to RViz
# --------------------------------------------------------------------------- #

# def publish_trajectory_to_rviz(states_np, frame_id="map"):
#     """
#     Publish the planned trajectory to RViz as both a Path and MarkerArray
#     """
#     # Create Path message
#     path_msg = Path()
#     path_msg.header = Header()
#     path_msg.header.stamp = rospy.Time.now()
#     path_msg.header.frame_id = frame_id
    
#     # Create MarkerArray for better visualization
#     marker_array = MarkerArray()
    
#     # Line strip marker for the trajectory
#     line_marker = Marker()
#     line_marker.header = path_msg.header
#     line_marker.ns = "trajectory_line"
#     line_marker.id = 0
#     line_marker.type = Marker.LINE_STRIP
#     line_marker.action = Marker.ADD
#     line_marker.scale.x = 0.05  # line width
#     line_marker.color = ColorRGBA(1.0, 0.0, 0.0, 1.0)  # red
    
#     # Arrow markers for heading
#     for i, state in enumerate(states_np):
#         x, y, v, theta = state
        
#         # Add to path
#         pose = PoseStamped()
#         pose.header = path_msg.header
#         pose.pose.position.x = float(x)
#         pose.pose.position.y = float(y)
#         pose.pose.position.z = 0.0
        
#         # Convert heading to quaternion
#         pose.pose.orientation.x = 0.0
#         pose.pose.orientation.y = 0.0
#         pose.pose.orientation.z = jnp.sin(theta / 2.0)
#         pose.pose.orientation.w = jnp.cos(theta / 2.0)
        
#         path_msg.poses.append(pose)
        
#         # Add point to line marker
#         point = Point()
#         point.x = float(x)
#         point.y = float(y)
#         point.z = 0.0
#         line_marker.points.append(point)
        
#         # Add arrow markers every few points for heading visualization
#         if i % 3 == 0:  # every 3rd point
#             arrow_marker = Marker()
#             arrow_marker.header = path_msg.header
#             arrow_marker.ns = "trajectory_arrows"
#             arrow_marker.id = i
#             arrow_marker.type = Marker.ARROW
#             arrow_marker.action = Marker.ADD
#             arrow_marker.pose = pose.pose
#             arrow_marker.scale.x = 0.2  # length
#             arrow_marker.scale.y = 0.05  # width
#             arrow_marker.scale.z = 0.05  # height
#             arrow_marker.color = ColorRGBA(0.0, 1.0, 0.0, 0.8)  # green
#             marker_array.markers.append(arrow_marker)
    
#     marker_array.markers.append(line_marker)
    
#     # Publish messages
#     path_pub.publish(path_msg)
#     marker_pub.publish(marker_array)
#     print("Published trajectory to RViz")

# publish_trajectory_to_rviz(states_np, frame_id="map")

# --------------------------------------------------------------------------- #
# 6.  Plots
# --------------------------------------------------------------------------- #
t = jnp.arange(T + 1) * dt

fig, axs = plt.subplots(3, 1, figsize=(8, 8))

# Get better state array using RK4
def get_states_from_RK4(states, controls, dt):
    def drag_force(v,
                Crr=0.015, m=MASS, g=GRAVITY,
                Fc=0.05, Fs=0.07, vs=0.08,
                c1=0.08, c2=0.02,
                eps=1e-3, delta=1e-3):
        """
        Returns longitudinal drag/friction force [N] opposing velocity v [m/s].
        Smooth & branchless; suitable for differentiable simulators.
        """
        sgn_smooth = jnp.tanh(v/eps)                  # smooth sign
        abs_smooth = jnp.sqrt(v*v + delta*delta)      # smooth |v|
        stribeck = Fc + (Fs - Fc) * jnp.exp(-(abs_smooth/vs)**2)
        rolling_plus_contact = Crr*m*g + stribeck
        return - sgn_smooth * rolling_plus_contact - c1*v - c2*v*abs_smooth
    # Define derivative function
    def f(t, state: jnp.ndarray) -> np.ndarray:
        """
        Computes the state derivative for the bicycle model.
        Follows the jitted dynamics function written above.
        Args:
            t: time step (used to index controls)
            state: state vector [x, y, v, θ]
        Returns:
            state_dot: state derivative vector [x_dot, y_dot, v_dot, θ_dot]
        """
        t = int(t)
        u_raw, delta_raw = controls[t]
        x, y, th, vx, vy, om = None, None, None, None, None, None
        # Unpack state
        if USING_FULL_DYNAMICS:
            x, y, th, vx, vy, om = state
        else:
            x, y, vx, th = state
        k = 100
        # clip controls
        #u_raw = jnp.tanh(u_raw)
        #u_raw = jnp.maximum(u_raw, 0.0) #ensure throttle is not too low
        #u_raw = U_L / (1 + U_A * jnp.exp(-U_B * u_raw)) - 1  # squashes throttle via sigmoid
        #u_raw = jnp.log(1 + jnp.exp(k * (u_raw))) / k  # squashes throttle via log
        #delta_raw += HEADING_BIAS_1
        #delta_raw = jnp.tanh(delta_raw)  # clamp steering to [-1, 1] via tanh

        state_dot = None
        if USING_FULL_DYNAMICS:
            ### NEW DYNAMICS ###
            # position and orientation
            c, s = jnp.cos(th), jnp.sin(th)
            x_dot = vx * c - vy * s
            y_dot = vx * s + vy * c
            th_dot = om

            # velocity in body frame and yaw rate
            slip_f = B_DELTA * delta_raw - (vy + COM_TO_FRONT * om) / vx
            slip_r = - (vy + COM_TO_REAR * om) / vx
            F_x = MASS * B_U_1 * u_raw
            F_y_f = -C_ALPHA_F * slip_f
            F_y_r = -C_ALPHA_R * slip_r
            vx_dot = (F_x / MASS) - om * vy
            vy_dot = (F_y_f + F_y_r) / MASS + om * vx
            om_dot = (COM_TO_FRONT * F_y_f - COM_TO_REAR * F_y_r) / YAW_INERTIA
            state_dot = jnp.array([x_dot, y_dot, th_dot, vx_dot, vy_dot, om_dot])
        else:
            ### OLD DYNAMICS ###
            # compute state changes
            x_dot = vx * jnp.cos(th)
            y_dot = vx * jnp.sin(th)
            #v_dot = B_U_1 * u_raw - drag_force(v) / MASS  # account for rolling friction
            vx_dot = B_U_1 * u_raw - ROLLING_FRICTION_COEFF * vx * vx / LENGTH  # account for rolling friction
            th_dot = vx / LENGTH * jnp.tan(B_DELTA * delta_raw)
            state_dot = jnp.array([x_dot, y_dot, vx_dot, th_dot])
        return state_dot
    
    # Runge-Kutta 4 integration
    t_values, y_values = runge_kutta_4(f, x0, 0.0, dt * T, dt)
    return y_values

better_states = get_states_from_RK4(states_np, controls_np, dt) if USING_RK4_GRAPHING else states_np

# (a) XY trajectory
axs[0].plot(better_states[:, 0], better_states[:, 1], marker="o", ms=2)
axs[0].set_title("Unicycle XY path")
axs[0].set_xlabel("x [m]")
axs[0].set_ylabel("y [m]")
axs[0].axis("equal")
axs[0].grid(alpha=0.3)

# Plot reference circle
center = (0,0)
radius = CIRCLE_RADIUS
theta_ref = jnp.linspace(0, 2 * jnp.pi, 100)
x_circle = center[0] + radius * jnp.cos(theta_ref)
y_circle = center[1] + radius * jnp.sin(theta_ref)
axs[0].plot(x_circle, y_circle, 'k--', label='Reference circle')
axs[0].legend()

# (b) heading + positions vs time
axs[1].plot(t, better_states[:, 0], label="x")
axs[1].plot(t, better_states[:, 1], label="y")
axs[1].plot(t, better_states[:, 2], label="v", linestyle="--")
axs[1].plot(t, better_states[:, 3], label="θ", linestyle="--")
axs[1].set_title("State trajectories"), axs[1].grid(alpha=0.3)
axs[1].legend(ncol=3)

# (c) controls
axs[2].step(t[:-1], controls_np[:, 0], where="post", label="u")
axs[2].step(t[:-1], controls_np[:, 1], where="post", label="δ")
axs[2].set_title("Control inputs")
axs[2].set_xlabel("time [s]")
axs[2].set_ylabel("command")
axs[2].grid(alpha=0.3)
axs[2].legend()

plt.tight_layout()
plt.savefig("figures/ex_5_ilqr.png", dpi=300)
plt.close()
