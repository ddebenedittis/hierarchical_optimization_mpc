#!/usr/bin/env python3

from ament_index_python.packages import get_package_share_directory
from casadi import *
from hierarchical_optimization_mpc.ho_mpc import HOMPC
import numpy as np


def main():
    # ======================== Define The System Model ======================= #
    
    # Define the state and input variables, and the discrete-time dynamics model.
    
    s = SX.sym('x', 3)  # state
    u = SX.sym('u', 2)  # input
    dt = 0.01           # timestep size
    
    # state_{k+1} = s_kpi(state_k, input_k)
    s_kp1 = vertcat(
        s[0] + dt * u[0] * cos(s[2]),
        s[1] + dt * u[0] * sin(s[2]),
        s[2] + dt * u[1]
    )
    
    # =========================== Define The Tasks =========================== #
    
    # Define the tasks separately.
    
    # Input limits
    v_max = 5
    v_min = -5
    omega_max = 1
    omega_min = -1
    
    task_input_limits = vertcat(
          u[0] - v_max,
        - u[0] + v_min,
          u[1] - omega_max,
        - u[1] + omega_min
    )
    task_input_limits_coeffs = [
        np.array([0, 0, 0, 0])
    ]
    
    task_vel_reference = vertcat(
        (s_kp1[0] - s[0]) / dt - 1,
        (s_kp1[1] - s[1]) / dt - 0
    )
        
    task_input_min = u
        
    # ============================ Create The MPC ============================ #
    
    hompc = HOMPC(s, u, s_kp1)
    hompc.n_control = 3
    hompc.n_pred = 0
                
    hompc.create_task(
        name = "input_limits", prio = 1,
        ineq_task_ls = task_input_limits,
        ineq_task_coeff=task_input_limits_coeffs
    )
        
    hompc.create_task(
        name = "reference", prio = 2,
        eq_task_ls = task_vel_reference,
    )
    
    hompc.create_task(
        name = "input_minimization", prio = 3,
        eq_task_ls = task_input_min,
    )
    
    # ======================================================================== #
        
    n_samples = 10000
    
    states = np.random.rand(n_samples, 3)
    states[:,2] = 2 * np.pi * states[:,2]
    
    inputs_star = np.zeros((n_samples, 2))
    zero_input = [np.zeros(2)] * hompc.n_control
        
    for k in range(n_samples):
        u_star = hompc(states[k, :], zero_input)
        for j in range(9):
            u_star = hompc()
        
        inputs_star[k, :] = u_star
        
        print(k)
        
    pkg_share_dir = get_package_share_directory('hompc_approximator')
        
    np.savetxt(pkg_share_dir + '/data/' + 'states.csv', states, delimiter=',')
    np.savetxt(pkg_share_dir + '/data/' + 'inputs_star.csv', inputs_star, delimiter=',')
        

if __name__ == '__main__':
    main()
