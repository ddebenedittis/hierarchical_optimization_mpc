#!/usr/bin/env python3

import copy
from dataclasses import dataclass
import time
import sys

import casadi as ca
import matplotlib as mpl
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np

from hierarchical_optimization_mpc.ho_mpc_multi_robot import HOMPCMultiRobot, QPSolver, TaskType
from hierarchical_optimization_mpc.disp_het_multi_rob import Animation, gen_arrow_head_marker, MultiRobotScatter


np.set_printoptions(
    precision=3,
    linewidth=300,
    suppress=True,
    threshold=sys.maxsize,
)


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
    time_start = time.time()
    
    np.random.seed()
    
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
        s[0][0] + dt * u[0][0] * ca.cos(s[0][2] + 1/2*dt * u[0][1]),
        s[0][1] + dt * u[0][0] * ca.sin(s[0][2] + 1/2*dt * u[0][1]),
        s[0][2] + dt * u[0][1]
    ))
    
    
    s.append(ca.SX.sym('x2', 2))     # state
    u.append(ca.SX.sym('u2', 2))     # input
    
    # state_{k+1} = s_kpi(state_k, input_k)
    s_kp1.append(ca.vertcat(
        s[1][0] + dt * u[1][0],
        s[1][1] + dt * u[1][1],
    ))
    
    n_robots = [2, 0]
    
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
    
    task_input_smooth = [
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
    
    task_input_smooth_coeffs = [
        [[np.array([0.9, 0.9, 0.8, 0.8])] for j in range(n_robots[0])],
        [[np.array([0.9, 0.9, 0.8, 0.8])] for j in range(n_robots[1])],
    ]
    
    # Centroid velocity reference
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
            u[0][1],
        ),
        ca.vertcat(
            u[1][0],
            - u[1][0],
            u[1][1],
            - u[1][1]
        ),
    ]
    
    aux = ca.SX.sym('aux', 2, 2)
    
    mapping = [
        ca.vertcat(
            s[0][0],
            s[0][1],
        ),
        ca.vertcat(
            s[1][0],
            s[1][1],
        ),
    ]
    
    # task_avoid_collision = ca.vertcat(
    #     (aux[0,0] - aux[1,0])**2 + (aux[0,1] - aux[1,1])**2 - 5,
    # )
    task_avoid_collision = ca.vertcat(
        (aux[0,0] - aux[1,0])**2 + (aux[0,1] - aux[1,1])**2 - 10,
    )
    
    # ============================ Create The MPC ============================ #
    
    hompc = HOMPCMultiRobot(s, u, s_kp1, n_robots, solver = QPSolver.quadprog)
    hompc.n_control = 2
    hompc.n_pred = 0
    
    hompc.create_task(
        name = "input_limits", prio = 1,
        type = TaskType.Same,
        ineq_task_ls = task_input_limits,
        # ineq_task_coeff = task_input_limits_coeffs,
    )
    
    hompc.create_task(
        name = "input_smooth", prio = 2,
        type = TaskType.SameTimeDiff,
        ineq_task_ls = task_input_smooth,
        ineq_task_coeff = task_input_smooth_coeffs,
    )
    
    hompc.create_task_bi(
        name = "collision_avoidance", prio = 3,
        type = TaskType.Bi,
        aux = aux,
        mapping = mapping,
        ineq_task_ls = task_avoid_collision,
    )
    
    hompc.create_task(
        name = "vel_ref", prio = 4,
        type = TaskType.Sum,
        eq_task_ls = task_vel_ref,
    )
    
    hompc.create_task(
        name = "input_minimization", prio = 5,
        type = TaskType.Same,
        eq_task_ls = task_input_min,
    )
    
    # ======================================================================== #
    
    s = [
        [np.multiply(np.random.random((3)), np.array([2, 2, 2*np.pi])) + np.array([-1, -1, 0])
         for _ in range(n_robots[0])],
        [np.multiply(np.random.random((2)), np.array([2, 2])) + np.array([-1, -1])
         for _ in range(n_robots[1])],
    ]
    
    n_steps = 1000
    
    s_history = [None] * n_steps
        
    for k in range(n_steps):
        print(k)
                
        u_star = hompc(copy.deepcopy(s))
        
        print(s)
        print(u_star)
        print()
                    
        s = evolve(s, u_star, dt)
                
        s_history[k] = copy.deepcopy(s)
        
    print(f"The time elapsed is {time.time() - time_start} seconds")
            
    fig, ax = plt.subplots()
    x = [np.zeros(n_r) for n_r in n_robots]
    y = [np.zeros(n_r) for n_r in n_robots]
    for c, n_r in enumerate(n_robots):
        for j in range(n_r):
            state = s[c][j]
            x[c][j] = state[0]
            y[c][j] = state[1]
    
    scat = MultiRobotScatter
    scat.unicycles = [None] * n_robots[0]
    scat.omnidir = [None] * n_robots[1]
        
    for i in range(n_robots[0]):
        scat.unicycles[i] = ax.scatter(x[0], y[0], 25, 'C0')
    for i in range(n_robots[1]):
        scat.omnidir[i] = ax.scatter(x[1], y[1], 25, 'C1')
        
    scat.centroid = ax.scatter(
        (np.mean(x[0])*n_robots[0] + np.mean(x[1])*n_robots[1]) / sum(n_robots),
        (np.mean(y[0])*n_robots[0] + np.mean(y[1])*n_robots[1]) / sum(n_robots),
        25, 'C2')
    
    ax.set(xlim=[-10., 10.], ylim=[-10., 10.], xlabel='x [m]', ylabel='y [m]')
    
    marker, scale = gen_arrow_head_marker(0)
    legend_elements = [
        Line2D([], [], marker=marker, markersize=20*scale, color='C0', linestyle='None', label='Unicycles'),
        Line2D([], [], marker='o', color='C1', linestyle='None', label='Omnidirectional Robot'),
        Line2D([], [], marker='o', color='C2', linestyle='None', label='Fleet Centroid'),
    ]
    
    ax.legend(handles=legend_elements)
    
    anim = Animation(scat, s_history)
    
    temp = FuncAnimation(fig=fig, func=anim.update, frames=range(n_steps), interval=30)
    plt.show()
    
    
if __name__ == '__main__':
    try:
        main()
    except:
        pass
