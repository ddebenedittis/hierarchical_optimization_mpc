import copy
import csv
import os
import time
from datetime import datetime
from itertools import combinations

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
import progressbar
from ament_index_python.packages import get_package_share_directory
from scipy.spatial.distance import pdist

from hierarchical_optimization_mpc.ho_mpc_multi_robot import (
    HOMPCMultiRobot,
    TaskBiCoeff,
    TaskType,
)
from hierarchical_optimization_mpc.tasks_creator_ho_mpc_mr import TasksCreatorHOMPCMultiRobot
from hierarchical_optimization_mpc.utils.disp_het_multi_rob import (
    MultiRobotArtistFlags,
    display_animation,
    plot_distances,
    save_snapshots,
)
from hierarchical_optimization_mpc.utils.robot_models import (
    RobCont,
    get_omnidirectional_model,
    get_unicycle_model,
)


def evolve(s: list[list[float]], u_star: list[list[float]], dt: float):
    n_intervals = 10
    # for j, _ in enumerate(s.omni):
    #     for _ in range(n_intervals):
    #         s.omni[j] = s.omni[j] + dt / n_intervals * np.array(
    #             [
    #                 u_star.omni[j][0] * np.cos(s.omni[j][2]),
    #                 u_star.omni[j][0] * np.sin(s.omni[j][2]),
    #                 u_star.omni[j][1],
    #             ]
    #         )

    # return s
    for j, _ in enumerate(s.omni):
        for _ in range(n_intervals):
            s.omni[j] = s.omni[j] + dt / n_intervals * np.array(
                [
                    u_star.omni[j][0],
                    u_star.omni[j][1],
                ]
            )

    return s


def main(n_robot):
    np.random.seed(1)
    n_steps = 500

    time_start = time.time()
    b = progressbar.ProgressBar(maxval=n_steps)
    b.start()
    # ============================== Parameters ============================= #

    dt = 0.05

    n_robots = RobCont(omni=n_robot)

    v_max = 2
    v_min = -2

    package_name = 'distributed_ho_mpc'
    workspace_dir = f'{get_package_share_directory(package_name)}/../../../..'
    out_dir = f'{workspace_dir}/out/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}-radial_switching_central_{n_robots.omni}A/'
    os.makedirs(out_dir, exist_ok=True)

    filename = f'{out_dir}/cntr_data.csv'
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)

        header = ['iter', 'Time']
        for i in range(n_robots.omni):
            header.append(f'stateX_{i}')
            header.append(f'stateY_{i}')
            # header.append(f'stateRHO_{i}')
            header.append(f'inputV_{i}')
            header.append(f'inputOH_{i}')

        writer.writerow(header)

    # ======================= Define The System Model ======================= #

    s = RobCont(omni=None)
    u = RobCont(omni=None)
    s_kp1 = RobCont(omni=None)

    # s.omni, u.omni, s_kp1.omni = get_unicycle_model(dt*10)
    s.omni, u.omni, s_kp1.omni = get_omnidirectional_model(10 * dt)

    # =========================== Define The Tasks ========================== #

    task_input_limits = RobCont(
        omni=ca.vertcat(
            u.omni[0] - v_max,
            -u.omni[0] - 2,  # v_min,
            u.omni[1] - 2,  # 1v_max,
            -u.omni[1] - 2,  # v_min
        )
    )
    # tasks_creator = TasksCreatorHOMPCMultiRobot(
    # s.tolist(),
    # u.tolist(),
    # s_kp1.tolist(),
    # dt,
    # n_robots.tolist(),
    # )
    # task_input_smooth, task_input_smooth_coeffs = tasks_creator.get_task_input_smooth()
    task_input_smooth = RobCont(
        omni=ca.vertcat(
            u.omni[0],
            -u.omni[0],
            u.omni[1],
            -u.omni[1],
        )
    ).tolist()

    task_input_smooth_coeffs = [
        [[np.array([0.95, 0.95, 0.9, 0.9])] for j in range(n_robots.omni)],
        [[]],
    ]
    # compute goals and starting points
    center = np.array([0, 0])  # choose the center
    num_points = n_robots.omni

    # radius from straight-line (chord) distance = 0.8
    radius = 1.5 / (2 * np.sin(np.pi / num_points))

    goals = []
    s_init = []

    for i in range(num_points):
        theta = 2 * np.pi * i / num_points
        x = center[0] + radius * np.cos(theta)
        y = center[1] + radius * np.sin(theta)
        goals.append(np.array([x, y]))

        theta = 2 * np.pi * i / num_points - np.pi
        x = center[0] + radius * np.cos(theta)
        y = center[1] + radius * np.sin(theta)
        s_init.append(np.array([x, y]))  # theta + np.pi

    task_pos_ref = []
    task_pos_ref_coeff = []
    for nn in range(n_robots.omni):
        task_pos_ref.append(RobCont(omni=ca.vertcat(s_kp1.omni[0], s_kp1.omni[1])))
        task_pos_ref_coeff.append(
            RobCont(
                omni=[[goals[nn]] for _ in range(n_robots.omni)],
            )
        )

    threshold = 0.6
    aux_avoid_collision = ca.SX.sym('aux', 2, 2)
    mapping_avoid_collision = RobCont(omni=ca.vertcat(s.omni[0], s.omni[1]))
    task_avoid_collision = ca.vertcat(
        -((aux_avoid_collision[0, 0] - aux_avoid_collision[1, 0]) ** 2)
        - (aux_avoid_collision[0, 1] - aux_avoid_collision[1, 1]) ** 2,
    )
    task_avoid_collision_coeff = [
        TaskBiCoeff(0, i, 0, j, 0, -(threshold**2))
        for i in range(n_robots.omni)
        for j in range(i + 1, n_robots.omni)
    ]

    # ============================ Create The MPC =========================== #

    hompc = HOMPCMultiRobot(
        s.tolist(),
        u.tolist(),
        s_kp1.tolist(),
        n_robots.tolist(),
    )
    hompc.n_control = 4
    hompc.n_pred = 0

    hompc.create_task(
        name='input_limits',
        prio=1,
        type=TaskType.Same,
        ineq_task_ls=task_input_limits.tolist(),
    )
    hompc.create_task(
        name='input_smooth',
        prio=2,
        type=TaskType.SameTimeDiff,
        ineq_task_ls=task_input_smooth,
        ineq_task_coeff=task_input_smooth_coeffs,
    )
    hompc.create_task_bi(
        name='collision_avoidance',
        prio=3,
        type=TaskType.Bi,
        aux=aux_avoid_collision,
        mapping=mapping_avoid_collision.tolist(),
        ineq_task_ls=task_avoid_collision,
        ineq_task_coeff=task_avoid_collision_coeff,
    )
    for nn in range(n_robots.omni):
        hompc.create_task(
            name=f'pos_ref_{nn}',
            prio=3,
            type=TaskType.Same,
            eq_task_ls=task_pos_ref[nn].tolist(),
            eq_task_coeff=task_pos_ref_coeff[nn].tolist(),
            # time_index=[2],
            robot_index=[[nn]],
        )

    # ======================================================================= #

    s = RobCont(omni=s_init)

    def agents_distance(state, pairwise_distances):
        """
        Plot the distance between the agents at each time step
        """
        positions_over_time = np.array(state)[:, :2]
        distances = pdist(positions_over_time, metric='euclidean')  # shape: (num_pairs,)
        for i, d in enumerate(distances):
            pairwise_distances[i].append(d)
        return pairwise_distances

    num_robots = n_robots.omni
    num_pairs = int(num_robots * (num_robots - 1) / 2)

    # Initialize one list per robot pair
    pairwise_distances = [[] for _ in range(num_pairs)]

    s_history = [None for _ in range(n_steps)]
    flags = MultiRobotArtistFlags()
    flags.voronoi = False
    # goals = np.array([[-5, -0], [-5, -5], [-5, 5], [5, -5]])
    gg = np.array(goals[: n_robots.omni])
    step_goal = 0
    for k in range(n_steps):
        if np.all(np.abs(np.array(s.omni)[:, :2] - gg) < 1e-2) and step_goal == 0:
            time_goal = time.time() - time_start
            step_goal = k
            break

        time_coord_start = time.time()

        u_star = hompc(copy.deepcopy(s.tolist()))

        # print(f's: {s}')
        # print(f'u_star: {u_star}')

        s = evolve(s, RobCont(omni=u_star[0]), dt)
        time_round = time_start - time.time()
        b.update(k + 1)
        with open(filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            row = [k, time_round]

            for i in range(n_robots.omni):
                row.extend(s.omni[i])
                row.extend(u_star[0][i])

            writer.writerow(row)
        s_history[k] = copy.deepcopy(s)

        last_step = k + 1

    time_elapsed = time.time() - time_start

    time_coord = time.time() - time_coord_start
    print(f'The time elapsed is {time_elapsed} seconds')
    print(f'The time elapsed for coordination is {time_coord} seconds')

    print('The time was used in the following phases:')
    max_key_len = max(map(len, hompc.solve_times.keys()))
    for key, value in hompc.solve_times.items():
        key_len = len(key)
        print(f'{key}: {" " * (max_key_len - key_len)}{value}')
        time_solution = value
    with open(f'{out_dir}time.txt', 'w') as file:
        file.write(
            f'dt: {dt}\n n_c: {4}\ntime elapsed: {time_elapsed}s\n time solution: {time_solution}'
        )

    # ========================= Visualization Options ======================== #

    # robot_pairs = list(combinations(range(num_robots), 2))
    # x = np.arange(1, last_step + 1) * dt
    # plt.figure(figsize=(10, 6))
    # for i, dist_list in enumerate(pairwise_distances):
    #     plt.plot(x, dist_list, label=f'Robots {robot_pairs[i]}')

    # plt.title('Time Evolution of Pairwise Robot Distances')
    # plt.xlabel('Time Step')
    # plt.ylabel('Distance')
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig(f'{out_dir}/distances.pdf', bbox_inches='tight', format='pdf')
    # plt.close()

    visual_method = 'none'

    s_history = [[[]] + s.tolist() for s in s_history[:last_step]]

    flags = MultiRobotArtistFlags()
    flags.voronoi = False
    goal = [
        [5, 5],
        [5, -5],
        [-5, -5],
        [-5, 5],
        [8, 3],
        [-8, 3],
        [8, -3],
        [-8, -3],
        [8, 0],
        [-8, 0],
        [0, 6],
        [0, -6],
    ]
    if visual_method is not None and visual_method != 'none':
        display_animation(
            s_history,
            s_history,
            goals,
            None,
            dt,
            visual_method,
            video_name=f'{out_dir}/video.mp4',
            x_lim=[-10, 10],
            y_lim=[-8, 8],
            flags=flags,
            n_c=4,
        )

    save_snapshots(
        s_history,
        goals,
        None,
        dt,
        [(last_step - 1) * dt],
        filename=f'{out_dir}/snapshot',
        x_lim=[-10, 10],
        y_lim=[-10, 10],
        flags=flags,
    )
    # plot_distances(
    #     s_history,
    #     0.05,  # dt
    #     0.6,  # dmin
    #     f'{out_dir}/distances.pdf',
    # )
    b.finish()
    return time_elapsed


if __name__ == '__main__':
    for i in range(3):
        n_robot = 10
        for i in range(4):
            main(n_robot)
            n_robot += 10
