#!/usr/bin/env python3

import copy
import sys
import time

import numpy as np

from hierarchical_optimization_mpc.auxiliary.evolve import evolve
from hierarchical_optimization_mpc.auxiliary.str2bool import str2bool
from hierarchical_optimization_mpc.ho_mpc_multi_robot import (
    HOMPCMultiRobot,
    QPSolver,
    TaskIndexes,
    TaskType,
)
from hierarchical_optimization_mpc.tasks_creator_ho_mpc_mr import TasksCreatorHOMPCMultiRobot
from hierarchical_optimization_mpc.utils.disp_het_multi_rob import (
    display_animation,
    plot_distances,
    save_snapshots,
)
from hierarchical_optimization_mpc.utils.robot_models import (
    get_omnidirectional_model,
    get_unicycle_model,
)

np.set_printoptions(
    precision=3,
    linewidth=300,
    suppress=True,
    threshold=sys.maxsize,
)


# ============================================================================ #
#                                 MAIN COVERAGE                                #
# ============================================================================ #


def main_coverage(
    hierarchical=True, n_robots=[5, 0], solver=QPSolver.quadprog, visual_method='plot'
):
    # ============================== Parameters ============================== #

    time_start = time.time()

    np.random.seed(0)

    # ======================== Define The System Model ======================= #

    # Define the state and input variables, and the discrete-time dynamics model.

    dt = 0.1  # timestep size

    s = [None for _ in range(2)]
    u = [None for _ in range(2)]
    s_kp1 = [None for _ in range(2)]

    s[0], u[0], s_kp1[0] = get_unicycle_model(dt)
    s[1], u[1], s_kp1[1] = get_omnidirectional_model(dt)

    # =========================== Define The Tasks =========================== #

    tasks_creator = TasksCreatorHOMPCMultiRobot(
        s,
        u,
        s_kp1,
        dt,
        n_robots,
    )
    tasks_creator.bounding_box = np.array([-20, 20, -20, 20])

    task_input_limits = tasks_creator.get_task_input_limits()

    task_input_smooth, task_input_smooth_coeffs = tasks_creator.get_task_input_smooth()

    task_coverage, task_coverage_coeff = tasks_creator.get_task_pos_ref(
        [[np.random.rand(2) for n_j in range(n_robots[c])] for c in range(len(n_robots))]
    )

    charging_stations = [
        [
            np.array([(2 * ((n_j // 2) % 2) - 1) * 19, (2 * (n_j % 2) - 1) * 19])
            for n_j in range(n_robots[c])
        ]
        for c in range(len(n_robots))
    ]
    task_charge, task_charge_coeff = tasks_creator.get_task_pos_ref(charging_stations)

    task_input_min = tasks_creator.get_task_input_min()

    # ============================ Create The MPC ============================ #

    hompc = HOMPCMultiRobot(s, u, s_kp1, n_robots, solver=solver, hierarchical=hierarchical)
    hompc.n_control = 4
    hompc.n_pred = 0

    hompc.create_task(
        name='input_limits',
        prio=1,
        type=TaskType.Same,
        ineq_task_ls=task_input_limits,
    )

    hompc.create_task(
        name='input_smooth',
        prio=2,
        type=TaskType.SameTimeDiff,
        ineq_task_ls=task_input_smooth,
        ineq_task_coeff=task_input_smooth_coeffs,
    )

    hompc.create_task(
        name='coverage',
        prio=4,
        type=TaskType.Same,
        eq_task_ls=task_coverage,
        eq_task_coeff=task_coverage_coeff,
        time_index=[0, 1, 2, 3],
    )

    hompc.create_task(
        name='input_minimization',
        prio=5,
        type=TaskType.Same,
        eq_task_ls=task_input_min,
    )

    hompc.create_task(
        name='charge',
        prio=6,
        type=TaskType.Same,
        eq_task_ls=task_charge,
        eq_task_coeff=task_charge_coeff,
        time_index=[0, 1, 2, 3],
        robot_index=[[0, 1], []],
    )

    # ======================================================================== #

    s = [
        [
            np.multiply(np.random.random((3)), np.array([10, 10, 2 * np.pi]))
            + np.array([-5, -5, 0])
            for _ in range(n_robots[0])
        ],
        [
            np.multiply(np.random.random((2)), np.array([2, 2])) + np.array([-1, -1])
            for _ in range(n_robots[1])
        ],
    ]

    print(s)

    n_steps = 250

    s_history = [None] * n_steps

    for k in range(n_steps):
        print(k)

        tasks_creator.states_bar = copy.deepcopy(s)
        cov_rob_idx = [[0, 1, 2, 3, 4, 5], []]
        task_coverage, task_coverage_coeff = tasks_creator.get_task_coverage(
            # cov_rob_idx
        )

        hompc.update_task(
            name='coverage',
            eq_task_ls=task_coverage,
            eq_task_coeff=task_coverage_coeff,
            # robot_index = cov_rob_idx,
        )

        if k == 150:
            hompc.update_task(
                name='charge',
                prio=3,
            )

        u_star = hompc(copy.deepcopy(s))

        print(f's: {s}')
        print(f'u_star: {u_star}')
        print()

        s = evolve(s, u_star, dt)

        s_history[k] = copy.deepcopy(s)

    # =========================== Print Time Usage =========================== #

    time_elapsed = time.time() - time_start
    print(f'The time elapsed is {time_elapsed} seconds')

    print('The time was used in the following phases:')
    max_key_len = max(map(len, hompc.solve_times.keys()))
    for key, value in hompc.solve_times.items():
        key_len = len(key)
        print(f'{key}: {" " * (max_key_len - key_len)}{value}')

    # ========================= Visualization Options ======================== #

    if visual_method is not None and visual_method != 'none':
        display_animation(
            s_history,
            charging_stations[0],
            None,
            dt,
            visual_method,
            show_trajectory=True,
            show_voronoi=True,
        )

    if visual_method == 'save':
        save_snapshots(
            s_history,
            charging_stations[0],
            None,
            dt,
            [0, 10, 25],
            'snapshot',
            show_trajectory=True,
            show_voronoi=True,
        )

    return time_elapsed


# ============================================================================ #
#                                MAIN_FORMATION                                #
# ============================================================================ #


def main_formation(
    hierarchical=True, n_robots=[5, 0], solver=QPSolver.quadprog, visual_method='plot'
):
    # ============================== Parameters ============================== #

    time_start = time.time()

    np.random.seed(0)

    # ======================== Define The System Model ======================= #

    # Define the state and input variables, and the discrete-time dynamics model.

    dt = 0.1  # timestep size

    s = [None for _ in range(2)]
    u = [None for _ in range(2)]
    s_kp1 = [None for _ in range(2)]

    s[0], u[0], s_kp1[0] = get_unicycle_model(dt)
    s[1], u[1], s_kp1[1] = get_omnidirectional_model(dt)

    # =========================== Define The Tasks =========================== #

    tasks_creator = TasksCreatorHOMPCMultiRobot(
        s,
        u,
        s_kp1,
        dt,
        n_robots,
    )
    tasks_creator.bounding_box = np.array([-20, 20, -20, 20])

    task_input_limits = tasks_creator.get_task_input_limits()

    task_input_smooth, task_input_smooth_coeffs = tasks_creator.get_task_input_smooth()

    obstacle_pos = np.array([10, 0.5])
    obstacle_size = 3
    task_obs_avoidance = tasks_creator.get_task_obs_avoidance(obstacle_pos, obstacle_size)

    (
        aux_avoid_collision,
        mapping_avoid_collision,
        task_avoid_collision,
        task_avoid_collision_coeff,
    ) = tasks_creator.get_task_avoid_collision(0.5)

    task_centroid_vel_ref = tasks_creator.get_task_centroid_vel_ref([1, 0])

    # task_vel_ref, task_vel_ref_coeff = tasks_creator.get_task_vel_ref(
    #     [3, 1]
    # )

    aux_formation, mapping_formation, task_formation, task_formation_coeff = (
        tasks_creator.get_task_formation()
    )

    task_input_min = tasks_creator.get_task_input_min()

    # ============================ Create The MPC ============================ #

    hompc = HOMPCMultiRobot(s, u, s_kp1, n_robots, solver=solver, hierarchical=hierarchical)
    hompc.n_control = 4
    hompc.n_pred = 0

    hompc.create_task(
        name='input_limits',
        prio=1,
        type=TaskType.Same,
        ineq_task_ls=task_input_limits,
    )

    hompc.create_task(
        name='input_smooth',
        prio=2,
        type=TaskType.SameTimeDiff,
        ineq_task_ls=task_input_smooth,
        ineq_task_coeff=task_input_smooth_coeffs,
    )

    hompc.create_task(
        name='obstacle_avoidance',
        prio=3,
        type=TaskType.Same,
        ineq_task_ls=task_obs_avoidance,
    )

    hompc.create_task_bi(
        name='collision_avoidance',
        prio=4,
        type=TaskType.Bi,
        aux=aux_avoid_collision,
        mapping=mapping_avoid_collision,
        ineq_task_ls=task_avoid_collision,
        ineq_task_coeff=task_avoid_collision_coeff,
    )

    hompc.create_task_bi(
        name='formation',
        prio=5,
        type=TaskType.Bi,
        aux=aux_formation,
        mapping=mapping_formation,
        eq_task_ls=task_formation,
        eq_task_coeff=task_formation_coeff,
    )

    hompc.create_task(
        name='centroid_vel_ref',
        prio=6,
        type=TaskType.Sum,
        eq_task_ls=task_centroid_vel_ref,
        time_index=TaskIndexes.Last,
    )

    # hompc.create_task(
    #     name = "vel_ref", prio = 5,
    #     type = TaskType.Same,
    #     eq_task_ls = task_vel_ref,
    #     eq_task_coeff = task_vel_ref_coeff,
    #     time_index = [0],
    # )

    hompc.create_task(
        name='input_minimization',
        prio=7,
        type=TaskType.Same,
        eq_task_ls=task_input_min,
    )

    # ======================================================================== #

    s = [
        [
            np.multiply(np.random.random((3)), np.array([10, 10, 2 * np.pi]))
            + np.array([-5, -5, 0])
            for _ in range(n_robots[0])
        ],
        [
            np.multiply(np.random.random((2)), np.array([2, 2])) + np.array([-1, 1.5])
            for _ in range(n_robots[1])
        ],
    ]

    print(s)

    n_steps = 200

    s_history = [None] * n_steps

    for k in range(n_steps):
        print(k)

        u_star = hompc(copy.deepcopy(s))

        print(f's: {s}')
        print(f'u_star: {u_star}')
        print()

        s = evolve(s, u_star, dt)

        s_history[k] = copy.deepcopy(s)

    # =========================== Print Time Usage =========================== #

    time_elapsed = time.time() - time_start
    print(f'The time elapsed is {time_elapsed} seconds')

    print('The time was used in the following phases:')
    max_key_len = max(map(len, hompc.solve_times.keys()))
    for key, value in hompc.solve_times.items():
        key_len = len(key)
        print(f'{key}: {" " * (max_key_len - key_len)}{value}')

    # ========================= Visualization Options ======================== #

    obstacles = np.concatenate((obstacle_pos, [obstacle_size]))
    if visual_method is not None and visual_method != 'none':
        display_animation(
            s_history,
            None,
            obstacles,
            dt,
            visual_method,
            show_trajectory=True,
            show_voronoi=False,
            x_lim=[-5, 20],
            y_lim=[-5, 15],
        )

    if visual_method == 'save':
        save_snapshots(
            s_history,
            None,
            obstacles,
            dt,
            [0, 18],
            'snapshot',
            show_trajectory=True,
            show_voronoi=False,
            x_lim=[-5, 20],
            y_lim=[-5, 15],
        )

        plot_distances(
            s_history,
            dt,
        )

    return time_elapsed


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Description')
    parser.add_argument(
        '--hierarchical',
        type=str2bool,
        metavar='bool',
        default=True,
        required=False,
        help='If True, uses the hierarchical approach',
    )
    parser.add_argument(
        '--n_robots',
        metavar='list[int]',
        default='[6,0]',
        required=False,
        help='Number of unicycles and omnidirectional robots (default [6,0])',
    )
    parser.add_argument(
        '--solver',
        metavar='{clarabel, osqp, proxqp, quadprog, reluqp}',
        default='quadprog',
        required=False,
        help='QP solver to use',
    )
    parser.add_argument(
        '--task',
        metavar='{coverage, com_ref_formation}',
        default='coverage',
        required=False,
        help='Type of task to solve',
    )
    parser.add_argument(
        '--visual_method',
        metavar='{plot, save, none}',
        default='plot',
        required=False,
        help='How to display the results',
    )
    args = parser.parse_args()

    if args.task == 'coverage':
        main_coverage(
            hierarchical=args.hierarchical,
            n_robots=[int(x) for x in args.n_robots.strip('[]').split(',') if x.strip().isdigit()],
            solver=args.solver,
            visual_method=args.visual_method,
        )
    elif args.task == 'com_ref_formation':
        main_formation(
            hierarchical=args.hierarchical,
            n_robots=[int(x) for x in args.n_robots.strip('[]').split(',') if x.strip().isdigit()],
            solver=args.solver,
            visual_method=args.visual_method,
        )
