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
dt = 0.5  # [s]  integration step
T = 30  # horizon length
n, m = 4, 2  # state & control sizes

MASS = 2.5  # kg
LENGTH = 0.26 # wheelbase
B_U = 4.5  # m/s² (≈0.2 g per unit throttle)
B_DELTA = 0.4601
THROTTLE_MAX = 0.5 # max throttle without slipping
HEADING_BIAS = 0.055  # steering bias
U_L = 1.5
U_A = 0.5
U_B = 4.1

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
    # clip controls
    u_raw = U_L / (1 + U_A * math.e ** (-U_B * u_raw)) - 1  # squashes throttle to a range
    #u_raw = u_raw * (1 - math.e ** ((-u_raw ** 2) / (SQUISH_RATE ** 2))) # squishes values near zero to nearer zero
    delta_raw += HEADING_BIAS

    # compute state changes
    x_dot = vx * jnp.cos(th)
    y_dot = vx * jnp.sin(th)
    vx_dot = B_U * u_raw
    th_dot = vx / LENGTH * jnp.tan(B_DELTA * delta_raw)

    # update state
    x_next = x + x_dot * dt
    y_next = y + y_dot * dt
    v_next = vx + vx_dot * dt
    th_next = th + th_dot * dt

    return jnp.array([x_next, y_next, v_next, th_next])


# -- single-step quadratic-ish cost -------------------------------------------------
w_pos, w_th, w_u = 1.0, 0.1, 1.0


def stage_cost(x, u):
    pos_cost = w_pos * (x[0] ** 2 + x[1] ** 2)
    head_cost = w_th * (x[2] ** 2)
    ctrl_cost = w_u * jnp.sum(u**2)
    return pos_cost + head_cost + ctrl_cost


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
        return pos + heading + control

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


cost = build_unicycle_cost(term_weight=100.0)

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
    for control in controls_np: # iterate over all controls
        curr_throttle = float(control[0])
        curr_steering = float(control[1])
        # clip controls if not already
        curr_throttle = min(1, max(-1, curr_throttle))
        curr_steering = min(1, max(-1, curr_steering))
        # send data over the network
        control_dict = {"throttle": curr_throttle, "steering": curr_steering}
        print("Unclipped control:", control)
        print("Clipped control:", control_dict)
        json_str = json.dumps(control_dict)
        client_socket.sendall(json_str.encode("utf-8"))
        # now, wait for dt number of ms to send the next command
        time.sleep(dt)

# --------------------------------------------------------------------------- #
# 4.  Plots
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
