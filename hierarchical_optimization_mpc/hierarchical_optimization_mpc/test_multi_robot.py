#!/usr/bin/env python3

import copy
import sys

import casadi as ca
from hierarchical_optimization_mpc.ho_mpc_multi_robot import HOMPCMultiRobot, TaskType
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np


np.set_printoptions(
    precision=3,
    linewidth=300,
    suppress=True,
    threshold=sys.maxsize,
)


class Animation():
    def __init__(self, scat, data) -> None:
        self.scat = scat
        self.data = data
    
    def update(self, frame):
        state = self.data[frame]
        counter = 0
        for s_c in state:
            for s_c_j in s_c:
                self.scat[counter].set_offsets((s_c_j[0], s_c_j[1]))
                counter += 1
        
        return self.scat
    

def evolve(s, u_star, dt):
    n_intervals = 10
    
    for c in range(len(s)):
        for j in range(len(s[c])):
            if c == 0:
                for _ in range(n_intervals):
                    s[c][j] = s[c][j] + dt / n_intervals * np.array([
                        u_star[c][j][0] * np.cos(s[c][j][2]),
                        u_star[c][j][0] * np.sin(s[c][j][2]),
                        u_star[c][j][1],
                    ])
            
            if c == 1:
                for _ in range(n_intervals):
                    s[c][j] = s[c][j] + dt / n_intervals * np.array([
                        u_star[c][j][0],
                        u_star[c][j][1],
                    ])
                    
    return s


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
    
    # Input limits
    v_max = 5
    v_min = -5
    omega_max = 1
    omega_min = -1
    
    task_input_limits = [
        ca.vertcat(
            u[0][0] - v_max,
            - u[0][0] + v_min,
            u[0][1] - omega_max,
            - u[0][1] + omega_min
        ),
        ca.vertcat(
            u[1][0] - v_max,
            - u[1][0] + v_min,
            u[1][1] - omega_max,
            - u[1][1] + omega_min
        ),
    ]
    
    task_input_limits_coeffs = [
        [np.array([0, 0, 0, 0])],
        [np.array([0, 0, 0, 0])],
    ]
    
    # Velocity reference
    task_vel_ref = [
        ca.vertcat(
            (s_kp1[0][0] - s[0][0]) / dt - 1,
            (s_kp1[0][1] - s[0][1]) / dt - 0,
        ) / sum(n_robots),
        ca.vertcat(
            (s_kp1[1][0] - s[1][0]) / dt - 1,
            (s_kp1[1][1] - s[1][1]) / dt - 0,
        ) / sum(n_robots),
    ]
    
    task_input_min = [
        ca.vertcat(
            u[0][0],
            - u[0][0],
            u[0][1],
            - u[0][1]
        ),
        ca.vertcat(
            u[1][0],
            - u[1][0],
            u[1][1],
            - u[1][1]
        ),
    ]
    
    # ============================ Create The MPC ============================ #
    
    hompc = HOMPCMultiRobot(s, u, s_kp1, n_robots)
    hompc.n_control = 3
    hompc.n_pred = 0
    
    hompc._initialize([
        [np.array([0, 0, 0])],
        [np.array([2, 0]), np.array([3,0])],
    ])
    
    hompc.create_task(
        name = "input_limits", prio = 1,
        type = TaskType.Same,
        ineq_task_ls = task_input_limits,
        ineq_task_coeff = task_input_limits_coeffs,
    )
    
    hompc.create_task(
        name = "vel_ref", prio = 2,
        type = TaskType.Sum,
        eq_task_ls = task_vel_ref,
    )
    
    hompc.create_task(
        name = "input_minimization", prio = 3,
        type = TaskType.Same,
        eq_task_ls = task_input_min,
    )
    
    # ======================================================================== #
    
    s = [
        [np.array([0, 0, 0.8])],
        [np.array([2, 0]), np.array([3,0])],
    ]
        
    n_steps = 100
    
    s_history = [None] * n_steps
        
    for k in range(n_steps):
        print(k)
                
        u_star = hompc(copy.deepcopy(s))
        
        print(s)
                    
        s = evolve(s, u_star, dt)
                
        s_history[k] = s
        
    fig, ax = plt.subplots()
    scat = []
    for c, n_r in enumerate(n_robots):
        for j in range(n_r):
            state = s[c][j]
            scat.append(ax.scatter(state[0], state[1]))
    
    ax.set(xlim=[-5, 5], ylim=[-5, 5], xlabel='x [m]', ylabel='y [m]')
    ax.legend()
    
    anim = Animation(scat, s_history)
    
    temp = FuncAnimation(fig=fig, func=anim.update, frames=range(n_steps), interval=30)
    plt.show()
    
    
if __name__ == '__main__':
    try:
        main()
    except:
        pass