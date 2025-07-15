#!/usr/bin/env python3

import sys
sys.path.append("./iLQR_jax_racing/iLQR") # needed to access iLQR modules in repo
import os, jax, argparse
import numpy as np

jax.config.update('jax_platform_name', 'cpu')
jax.config.update('jax_enable_x64', True)

from iLQR.utils import *
from iLQR.ilqr import iLQR
from iLQR.shielding import ILQshielding, NaiveSwerving
from iLQR.ellipsoid_obj import EllipsoidObj

# imports for TCP
import socket
import json

HOST = "192.168.50.236"
PORT = 65432
DT = 0.5
RECV_CODE = "get_pose"


def main(config_file):
  # Loads the config and track file.
  config = load_config(config_file)
  track = load_track(config)

  # Constructs static obstacles.
  static_obs_list = []
  obs_a = config.LENGTH / 2.0
  obs_b = config.WIDTH / 2.0

  obs_q1 = np.array([0, 5.6])[:, np.newaxis]
  obs_Q1 = np.diag([obs_a**2, obs_b**2])
  static_obs1 = EllipsoidObj(q=obs_q1, Q=obs_Q1)
  static_obs_list.append([static_obs1 for _ in range(config.N)])

  obs_q2 = np.array([-2.15, 4.0])[:, np.newaxis]
  obs_Q2 = np.diag([obs_b**2, obs_a**2])
  static_obs2 = EllipsoidObj(q=obs_q2, Q=obs_Q2)
  static_obs_list.append([static_obs2 for _ in range(config.N)])

  obs_q3 = np.array([-2.4, 2.0])[:, np.newaxis]
  obs_Q3 = np.diag([obs_b**2, obs_a**2])
  static_obs3 = EllipsoidObj(q=obs_q3, Q=obs_Q3)
  static_obs_list.append([static_obs3 for _ in range(config.N)])

  static_obs_heading_list = [-np.pi, -np.pi / 2, -np.pi / 2]

  # Connects to TCP server on Jetson - all code will have socket access now.
  with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
    client_socket.connect((HOST, PORT))

    # Initialization.
    solver = iLQR(track, config, safety=False)  # Nominal racing iLQR
    solver_sh = iLQR(track, config, safety=True)  # Shielding safety backup iLQR

    shielding = ILQshielding(
        config, solver_sh, [static_obs1, static_obs2, static_obs3], static_obs_list, N_sh=15
    )

    pos0, psi0 = track.interp([2])  # Position and yaw on the track.
    x_cur = np.array([3.3, 4.0, 0, np.pi / 2])  # Initial state (x, y, v, psi)
    # x_cur = np.array([pos0[0], pos0[1], 0, psi0[-1]])
    init_control = np.zeros((2, config.N))
    curr_throttle = 0, curr_steering = 0
    t_total = 0.

    itr_receding = config.MAX_ITER_RECEDING  # The number of receding iterations.
    state_hist = np.zeros((4, itr_receding))
    control_hist = np.zeros((2, itr_receding))
    plan_hist = np.zeros((6, config.N, itr_receding))
    K_hist = np.zeros((2, 4, config.N - 1, itr_receding))
    fx_hist = np.zeros((4, 4, config.N, itr_receding))
    fu_hist = np.zeros((4, 2, config.N, itr_receding))

    # Define disturbances.
    sigma = np.array([config.SIGMA_X, config.SIGMA_Y, config.SIGMA_V, config.SIGMA_THETA])

    # iLQR Planning.
    for i in range(itr_receding):
      # Plans the trajectory using iLQR.
      states, controls, t_process, status, _, K_closed_loop, fx, fu = (
          solver.solve(x_cur, controls=init_control, obs_list=static_obs_list)
      )

      # Shielding.
      control_sh = shielding.run(x=x_cur, u_nominal=controls[:, 0])

      # Executes the control.
      # using sim:
      #x_cur = solver.dynamics.forward_step(x_cur, control_sh, step=1, noise=sigma)[0]
      #print("[{}]: solver returns status {} and uses {:.3f}.".format(i, status, t_process), end='\r')

      # executing on hardware over tcp:
      accel = control_sh[0]
      delta = control_sh[1]
      # adjust throttle and steering according to controls
      curr_throttle += accel
      curr_throttle = min(1, max(-1, curr_throttle))
      curr_steering += delta
      curr_steering = min(1, max(-1, curr_steering))
      # send data over the network
      control_dict = {"throttle": curr_throttle, "steering": curr_steering}
      json_str = json.dumps(control_dict)
      client_socket.sendall(json_str.encode("utf-8"))
      if i > 0:  # Excludes JAX compilation time at the first time step.
        t_total += t_process
      
      # now, get current state since there may be discrepancies irl
      client_socket.sendall(RECV_CODE.encode("utf-8")) # sends message that we want current pose info
      raw_pose_data = client_socket.recv(1024).decode()
      state_dict = json.loads(raw_pose_data)
      x_cur = np.ndarray([state_dict["x"], state_dict["y"], state_dict["v"], state_dict["psi"]])

      # Records planning history, states and controls.
      plan_hist[:4, :, i] = states
      plan_hist[4:, :, i] = controls
      state_hist[:, i] = states[:, 0]
      control_hist[:, i] = controls[:, 0]

      K_hist[:, :, :, i] = K_closed_loop
      fx_hist[:, :, :, i] = fx
      fu_hist[:, :, :, i] = fu

      # Updates the nominal control signal for warmstart of next receding horizon.
      init_control[:, :-1] = controls[:, 1:]

    print("Planning uses {:.3f}.".format(t_total))


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "-cf", "--config_file", help="config file path", type=str,
      default=os.path.join("", "example_racecar.yaml")
  )
  args = parser.parse_args()
  main(args.config_file)
