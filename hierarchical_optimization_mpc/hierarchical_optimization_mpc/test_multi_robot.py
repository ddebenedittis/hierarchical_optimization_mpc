#!/usr/bin/env python3

import casadi as ca
from hierarchical_optimization_mpc.ho_mpc_multi_robot import HOMPCMultiRobot
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np


np.set_printoptions(
    precision=3,
    linewidth=300,
    suppress=True
)


class Animation():
    def __init__(self, scat, data) -> None:
        self.scat = scat
        self.data = data
    
    def update(self, frame):
        self.scat.set_offsets((self.data[frame, 0], self.data[frame, 1]))
        return self.scat


def main():
    # ======================== Define The System Model ======================= #
    
    # Define the state and input variables, and the discrete-time dynamics model.
    
    s = []
    u = []
    dt = 0.01           # timestep size
    s_kp1 = []
    
    s.append(ca.SX.sym('x', 3))     # state
    u.append(ca.SX.sym('u', 2))     # input
    
    # state_{k+1} = s_kpi(state_k, input_k)
    s_kp1.append(ca.vertcat(
        s[0][0] + dt * u[0][0] * ca.cos(s[0][2]),
        s[0][1] + dt * u[0][0] * ca.sin(s[0][2]),
        s[0][2] + dt * u[0][1]
    ))
    
    
    s.append(ca.SX.sym('x2', 2))     # state
    u.append(ca.SX.sym('u2', 2))     # input
    
    # state_{k+1} = s_kpi(state_k, input_k)
    s_kp1.append(ca.vertcat(
        s[1][0] + dt * u[1][0],
        s[1][1] + dt * u[1][1],
    ))
    
    n_robots = [1, 2]
    
    # =========================== Define The Tasks =========================== #
    

    
    # ============================ Create The MPC ============================ #
    
    hompc = HOMPCMultiRobot(s, u, s_kp1, n_robots)
    hompc.n_control = 3
    hompc.n_pred = 0
    
    hompc._initialize([
        [np.array([0, 0, 0])],
        [np.array([2, 0]), np.array([3,0])],
    ])
    
    hompc()
    
    # hompc.create_task(
    #     name = "input_limits", prio = 1,
    #     ineq_task_ls = task_input_limits,
    #     ineq_task_coeff=task_input_limits_coeffs
    # )
        
    # hompc.create_task(
    #     name = "reference", prio = 2,
    #     eq_task_ls = task_vel_reference,
    # )
    
    # hompc.create_task(
    #     name = "input_minimization", prio = 3,
    #     eq_task_ls = task_input_min,
    # )
    
    # ======================================================================== #
    
    # s = np.array([0., 0., 0.8])
    
    # n_steps = 1000
    
    # s_history = np.zeros((n_steps, 3))
        
    # for k in range(n_steps):
    #     print(k)
        
    #     u_star = hompc(s)
                
    #     for i in range(10):
    #         s = s + dt / 10 * np.array([
    #             u_star[0] * np.cos(s[2]),
    #             u_star[0] * np.sin(s[2]),
    #             u_star[1]
    #         ])
        
    #     print(s)
    #     print(u_star)
    #     print()
        
    #     s_history[k, :] = s
        
    # fig, ax = plt.subplots()
    # scat = ax.scatter(s[0], s[1])
    # ax.set(xlim=[-5, 5], ylim=[-5, 5], xlabel='x [m]', ylabel='y [m]')
    # ax.legend()
    
    # anim = Animation(scat, s_history)
    
    # temp = FuncAnimation(fig=fig, func=anim.update, frames=range(1000), interval=30)
    # plt.show()
    
    
if __name__ == '__main__':
    main()
