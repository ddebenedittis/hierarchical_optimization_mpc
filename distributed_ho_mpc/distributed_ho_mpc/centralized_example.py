import copy
import time

import casadi as ca
import numpy as np

from hierarchical_optimization_mpc.ho_mpc_multi_robot import HOMPCMultiRobot, TaskIndexes, QPSolver, TaskBiCoeff, TaskType
from hierarchical_optimization_mpc.tasks_creator_ho_mpc_mr import TasksCreatorHOMPCMultiRobot
from hierarchical_optimization_mpc.utils.disp_het_multi_rob import (
    display_animation,
    MultiRobotArtists,
    plot_distances,
    save_snapshots
)
from hierarchical_optimization_mpc.utils.robot_models import get_omnidirectional_model, RobCont


def evolve(s: list[list[float]], u_star: list[list[float]], dt: float):
    n_intervals = 10
    
    for j, _ in enumerate(s.omni):
        
        for _ in range(n_intervals):
            s.omni[j] = s.omni[j] + dt / n_intervals * np.array([
                u_star.omni[j][0],
                u_star.omni[j][1],
            ])
    
    return s


def main():
    
    np.random.seed(1)
    
    time_start = time.time()
    
    # ============================== Parameters ============================= #
    
    dt = 0.1
    
    n_robots = RobCont(omni=4)
    
    v_max = 5
    v_min = -5
    
    # ======================= Define The System Model ======================= #
    
    s = RobCont(omni=None)
    u = RobCont(omni=None)
    s_kp1 = RobCont(omni=None)
    
    s.omni, u.omni, s_kp1.omni = get_omnidirectional_model(dt)
    
    # =========================== Define The Tasks ========================== #
    
    task_input_limits = RobCont(omni=ca.vertcat(
          u.omni[0] - v_max,
        - u.omni[0] + v_min,
          u.omni[1] - v_max,
        - u.omni[1] + v_min
    ))
    
    # ======================================================================= #
    
    task_pos_ref_1 = RobCont(omni=ca.vertcat(s_kp1.omni[0], s_kp1.omni[1]))
    task_pos_ref_1_coeff = RobCont(
        omni=[[np.array([-10, 5])] for _ in range(n_robots.omni)],
    )
    
    # ======================================================================= #
    
    aux = ca.SX.sym('aux', 2, 2)
    mapping = RobCont(omni=ca.vertcat(s.omni[0], s.omni[1]))
    task_formation = ca.vertcat(
        (aux[0,0] - aux[1,0])**2 + (aux[0,1] - aux[1,1])**2 - 0,
    )
    task_formation_coeff = [
        TaskBiCoeff(0, 0, 0, 1, 0, 5**2),
        TaskBiCoeff(0, 2, 0, 3, 0, 5**2),
    ]
    
    # ======================================================================= #
    
    task_pos_ref_2 = RobCont(omni=ca.vertcat(s_kp1.omni[0], s_kp1.omni[1]))
    task_pos_ref_2_coeff = RobCont(
        omni=[[np.array([-10, 5])] for _ in range(n_robots.omni)]
    )
    
    # ======================================================================= #
    
    task_pos_ref_3 = RobCont(omni=ca.vertcat(s_kp1.omni[0], s_kp1.omni[1]))
    task_pos_ref_3_coeff = RobCont(
        omni=[[np.array([10, -5])] for _ in range(n_robots.omni)]
    )
    
    # ======================================================================= #
    
    task_input_min = RobCont(omni=ca.vertcat(u.omni[0], u.omni[1]))
    
    # ============================ Create The MPC =========================== #
    
    hompc = HOMPCMultiRobot(
        s.tolist(),
        u.tolist(),
        s_kp1.tolist(),
        n_robots.tolist(),
    )
    hompc.n_control = 1
    hompc.n_pred = 0
    
    # hompc.create_task(
    #     name="input_limits", prio=1,
    #     type=TaskType.Same,
    #     ineq_task_ls=task_input_limits.tolist(),
    # )
    # hompc.create_task(
    #     name="pos_ref_1", prio=2,
    #     type=TaskType.Same,
    #     eq_task_ls=task_pos_ref_1.tolist(),
    #     eq_task_coeff=task_pos_ref_1_coeff.tolist(),
    #     robot_index=[[0]],
    # )
    hompc.create_task_bi(
        name="formation", prio=3,
        type=TaskType.Bi,
        aux=aux,
        mapping=mapping.tolist(),
        eq_task_ls=task_formation,
        eq_task_coeff=task_formation_coeff,
    )
    # hompc.create_task(
    #     name="pos_ref_2", prio=4,
    #     type=TaskType.Same,
    #     eq_task_ls=task_pos_ref_2.tolist(),
    #     eq_task_coeff=task_pos_ref_2_coeff.tolist(),
    #     robot_index=[[1]]
    # )
    # hompc.create_task(
    #     name="pos_ref_3", prio=5,
    #     type=TaskType.Same,
    #     eq_task_ls=task_pos_ref_3.tolist(),
    #     eq_task_coeff=task_pos_ref_3_coeff.tolist(),
    #     robot_index=[[2]]
    # )
    # hompc.create_task(
    #     name="input_min", prio=6,
    #     type=TaskType.Same,
    #     eq_task_ls=task_input_min.tolist(),
    # )
    
    # ======================================================================= #
    
    s = RobCont(omni=
        [np.multiply(np.random.random((2)), np.array([2, 2])) + np.array([-1, -1])
         for _ in range(n_robots.omni)],
    )
    
    n_steps = 100
    
    s_history = [None for _ in range(n_steps)]
    
    for k in range(n_steps):
        print(k)
        
        u_star = hompc(copy.deepcopy(s.tolist()))
        
        print(f"s: {s}")
        print(f"u_star: {u_star}")
        print()
        
        s = evolve(s, RobCont(omni=u_star[0]), dt)
        
        s_history[k] = copy.deepcopy(s)
        
    time_elapsed = time.time() - time_start
    print(f"The time elapsed is {time_elapsed} seconds")
    
    print( "The time was used in the following phases:")
    max_key_len = max(map(len, hompc.solve_times.keys()))
    for key, value in hompc.solve_times.items():
        key_len = len(key)
        print(f"{key}: {' '*(max_key_len-key_len)}{value}")
    
    # ========================= Visualization Options ======================== #
    
    visual_method = 'plot'
    charging_stations = [None]
    
    s_history = [s.tolist() for s in s_history]
    
    artist_flags = MultiRobotArtists(
        centroid=True, goals=True, obstacles=False,
        past_trajectory=True,
        robots=RobCont(omni=True),
        robot_names=True,
        voronoi=False,
    )
    
    if visual_method is not None and visual_method != 'none':
        display_animation(
            s_history, charging_stations[0], None, dt, visual_method,
            artist_flags,
        )
        
    if visual_method == 'save':
        save_snapshots(
            s_history, charging_stations[0], None, dt, [0, 10, 25], 'snapshot',
            artist_flags,
        )
        
    return time_elapsed
    

if __name__ == '__main__':
    main()
