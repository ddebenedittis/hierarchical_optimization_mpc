#!/usr/bin/env python3

import copy
import time
import sys

import casadi as ca
import numpy as np

from hierarchical_optimization_mpc.ho_mpc_multi_robot import HOMPCMultiRobot, TaskIndexes, QPSolver, TaskBiCoeff, TaskType
from hierarchical_optimization_mpc.disp_het_multi_rob import display_animation
from hierarchical_optimization_mpc.tasks_creator_ho_mpc_mr import TasksCreatorHOMPCMultiRobot
from hierarchical_optimization_mpc.voronoi_task import BoundedVoronoi


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


# ============================================================================ #
#                                     MAIN                                     #
# ============================================================================ #

def main():
    time_start = time.time()
    
    np.random.seed(3)
    
    # ======================== Define The System Model ======================= #
    
    # Define the state and input variables, and the discrete-time dynamics model.
    
    s = []
    u = []
    dt = 0.1           # timestep size
    s_kp1 = []
    
    s.append(ca.SX.sym('x', 3))     # state
    u.append(ca.SX.sym('u', 2))     # input
    
    # state_{k+1} = s_kpi(state_k, input_k)
    s_kp1.append(ca.vertcat(
        s[0][0] + dt * u[0][0] * ca.cos(s[0][2] + 0/2*dt * u[0][1]),
        s[0][1] + dt * u[0][0] * ca.sin(s[0][2] + 0/2*dt * u[0][1]),
        s[0][2] + dt * u[0][1]
    ))
    
    
    s.append(ca.SX.sym('x2', 2))     # state
    u.append(ca.SX.sym('u2', 2))     # input
    
    # state_{k+1} = s_kpi(state_k, input_k)
    s_kp1.append(ca.vertcat(
        s[1][0] + dt * u[1][0],
        s[1][1] + dt * u[1][1],
    ))
    
    n_robots = [6, 0]
    
    # =========================== Define The Tasks =========================== #
    
    tasks_creator = TasksCreatorHOMPCMultiRobot(
        s, u, s_kp1, dt, n_robots,
    )
    
    task_input_limits = tasks_creator.get_task_input_limits()
    
    task_input_smooth, task_input_smooth_coeffs = tasks_creator.get_task_input_smooth()
    
    # task_centroid_vel_ref = tasks_creator.get_task_centroid_vel_ref([3, 1])
    
    # task_vel_ref, task_vel_ref_coeff = tasks_creator.get_task_vel_ref(
    #     [3, 1]
    # )
    
    task_pos_ref, task_pos_ref_coeff = tasks_creator.get_task_pos_ref(
        [[np.random.rand(2) for n_j in range(n_robots[c])] for c in range(len(n_robots))]
    )
    
    # aux, mapping, task_formation, task_formation_coeff = tasks_creator.get_task_formation()
    
    task_input_min = tasks_creator.get_task_input_min()
    
    # ============================ Create The MPC ============================ #
    
    hompc = HOMPCMultiRobot(s, u, s_kp1, n_robots, solver = QPSolver.quadprog)
    hompc.n_control = 4
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
    
    # hompc.create_task_bi(
    #     name = "collision_avoidance", prio = 3,
    #     type = TaskType.Bi,
    #     aux = aux,
    #     mapping = mapping,
    #     eq_task_ls = task_avoid_collision,
    #     eq_task_coeff = task_avoid_collision_coeff,
    # )
    
    # hompc.create_task_bi(
    #     name = "formation", prio = 3,
    #     type = TaskType.Bi,
    #     aux = aux,
    #     mapping = mapping,
    #     eq_task_ls = task_formation,
    #     eq_task_coeff = task_formation_coeff,
    # )
    
    # hompc.create_task(
    #     name = "centroid_vel_ref", prio = 4,
    #     type = TaskType.Sum,
    #     eq_task_ls = task_centroid_vel_ref,
    #     time_index = TaskIndexes.Last,
    # )
    
    # hompc.create_task(
    #     name = "vel_ref", prio = 4,
    #     type = TaskType.Same,
    #     eq_task_ls = task_vel_ref,
    #     eq_task_coeff = task_vel_ref_coeff,
    #     time_index = [0],
    # )
    
    hompc.create_task(
        name = "pos_ref", prio = 4,
        type = TaskType.Same,
        eq_task_ls = task_pos_ref,
        eq_task_coeff = task_pos_ref_coeff,
        time_index = [0, 1, 2, 3],
    )
    
    hompc.create_task(
        name = "input_minimization", prio = 5,
        type = TaskType.Same,
        eq_task_ls = task_input_min,
    )
    
    # ======================================================================== #
    
    s = [
        [np.multiply(np.random.random((3)), np.array([10, 10, 2*np.pi])) + np.array([-5, -5, 0])
         for _ in range(n_robots[0])],
        [np.multiply(np.random.random((2)), np.array([2, 2])) + np.array([-1, -1])
         for _ in range(n_robots[1])],
    ]
    
    print(s)
    
    n_steps = 200
    
    s_history = [None] * n_steps
            
    for k in range(n_steps):
        print(k)
        
        towers = np.array(
            [e[0:2] for e in s[0]]
        )
        bounding_box = np.array([-20, 20, -20, 20])
        b_vor = BoundedVoronoi(towers, bounding_box)
        centroids = b_vor.get_centroids_2()
        pos_ref = [[np.array([0, 0]) for _ in range(n_robots[0])]]
        for i in range(len(pos_ref[0])):
            pos_ref[0][i] = centroids[i,:]
            
        task_pos_ref, task_pos_ref_coeff = tasks_creator.get_task_pos_ref(
            pos_ref
        )
        
        hompc.update_task(
            name = "pos_ref",
            eq_task_ls = task_pos_ref,
            eq_task_coeff = task_pos_ref_coeff,
        )
        
        u_star = hompc(copy.deepcopy(s))
        
        print(s)
        print(u_star)
        print()
        
        s = evolve(s, u_star, dt)
        
        s_history[k] = copy.deepcopy(s)
        
    print(f"The time elapsed is {time.time() - time_start} seconds")
    
    display_animation(s_history, dt)
    
    
if __name__ == '__main__':
    try:
        main()
    except:
        pass
