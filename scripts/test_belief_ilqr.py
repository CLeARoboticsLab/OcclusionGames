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
import matplotlib.pyplot as plt
import time

# --------------------------------------------------------------------------- #
# 0.  Import your solver (change path if needed)
# --------------------------------------------------------------------------- #
import sys
sys.path.append("C:\\Users\\jaivn\\Desktop\\OcclusionGameRepo")
from belief_ilqr_solver.solver import iLQR  # ← keep consistent with your repo


# --------------------------------------------------------------------------- #
# 1.  Dynamics & cost
# --------------------------------------------------------------------------- #
dt = 0.1  # [s]  integration step
T = 30  # horizon length
n, m = 3, 2  # state & control sizes


@jax.jit
def dynamics(state: jnp.ndarray, control: jnp.ndarray) -> jnp.ndarray:
    """
    Unicycle discrete-time dynamics
    state   = [x, y, θ]
    control = [v, ω]      (linear & angular velocity commands)
    """
    x, y, th = state
    v, w = control

    x_next = x + dt * v * jnp.cos(th)
    y_next = y + dt * v * jnp.sin(th)
    th_next = th + dt * w
    return jnp.array([x_next, y_next, th_next])


# -- single-step quadratic-ish cost -------------------------------------------------
w_pos, w_th, w_u = 1.0, 0.1, 1e-2


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

# --------------------------------------------------------------------------- #
# 2.  Problem setup
# --------------------------------------------------------------------------- #
x0 = jnp.array([-6.0, 10.0, 0.9])  # initial state
u_init = jnp.zeros((T, m))  # zero-controls guess

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
