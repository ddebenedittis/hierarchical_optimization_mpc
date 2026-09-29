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
from hierarchical_optimization_mpc.utils.disp_het_multi_rob import (
    MultiRobotArtistFlags,
    display_animation,
    save_snapshots,
)
from hierarchical_optimization_mpc.utils.robot_models import RobCont, get_unicycle_model


def evolve(s: list[list[float]], u_star: list[list[float]], dt: float):
    n_intervals = 10

    for j, _ in enumerate(s.omni):
        for _ in range(n_intervals):
            s.omni[j] = s.omni[j] + dt / n_intervals * np.array(
                [
                    u_star.omni[j][0] * np.cos(s.omni[j][2]),
                    u_star.omni[j][0] * np.sin(s.omni[j][2]),
                    u_star.omni[j][1],
                ]
            )

    return s


def main(num_robots, steps):
    np.random.seed(1)
    b = progressbar.ProgressBar(maxval=steps)
    b.start()

    package_name = 'distributed_ho_mpc'
    workspace_dir = f'{get_package_share_directory(package_name)}/../../../..'
    out_dir = f'{workspace_dir}/out/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}-narrow_gap_cntr_{num_robots}A/'
    os.makedirs(out_dir, exist_ok=True)

    time_start = time.time()

    # ============================== Parameters ============================= #

    dt = 0.05

    n_robots = RobCont(omni=num_robots)

    v_max = 2.0
    v_min = -2.0

    # ======================= Define The System Model ======================= #

    s = RobCont(omni=None)
    u = RobCont(omni=None)
    s_kp1 = RobCont(omni=None)

    s.omni, u.omni, s_kp1.omni = get_unicycle_model(dt * 2)

    time_start = time.time()

    # n_robots = num_robots

    # --- obstacles ---
    obs_centers = [np.array([0.0, 8.0]), np.array([0.0, -8.0])]

    R_obs = 6 - 0.5 * (n_robots.omni // 10)

    s_init = []
    goals = []
    placed = 0
    # radius from straight-line (chord) distance = 0.8

    line_n = [6, 10, 6, 3]
    line = [3.5, 5, 6.5, 8]
    h = [2.5, 2, 3, 1]

    while placed != n_robots.omni:
        for j in range(len(line_n)):
            if placed == n_robots.omni:
                continue
            for i in range(line_n[j]):
                if placed == n_robots.omni:
                    continue
                goals.append(np.array([line[len(line) - j - 1], (h[j] - i * 1.2)]))
                goals.append(np.array([-line[len(line) - j - 1], (h[j] - i * 1.2)]))
                s_init.append(np.array([-line[j], (h[j] - i * 1.2), 0]))
                s_init.append(np.array([line[j], (h[j] - i * 1.2), np.pi]))
                placed += 2

    # ========================== Prepare The Data ========================== #
    filename = f'{out_dir}/cntr_data.csv'
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)

        header = ['k', 'Time']
        for i in range(n_robots.omni):
            header.append(f'stateX_{i}')
            header.append(f'stateY_{i}')
            header.append(f'stateTheta_{i}')
            header.append(f'inputV_{i}')
            header.append(f'inputOH_{i}')

        writer.writerow(header)

    # =========================== Define The Tasks ========================== #

    task_input_limits = RobCont(
        omni=ca.vertcat(
            u.omni[0] - v_max,
            -u.omni[0] + 0,  # v_min,
            u.omni[1] - 2.0,  # 1v_max,
            -u.omni[1] - 2.0,  # v_min
        )
    )

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

    # ===============================
    obstacle_pos_1 = np.array([0, obs_centers[0][1]])
    obstacle_pos_2 = np.array([0, obs_centers[1][1]])
    obstacle_size = R_obs
    task_obs_avoidance = [None, None]
    task_obs_avoidance[0] = [
        ca.vertcat(
            -((s.omni[0] - obstacle_pos_1[0]) ** 2)
            - (s.omni[1] - obstacle_pos_1[1]) ** 2
            + obstacle_size**2
        )
    ]
    task_obs_avoidance[1] = [
        ca.vertcat(
            -((s.omni[0] - obstacle_pos_2[0]) ** 2)
            - (s.omni[1] - obstacle_pos_2[1]) ** 2
            + obstacle_size**2
        )
    ]

    # ======================================================================= #

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
    hompc.n_control = 3
    hompc.n_pred = 0

    task_pos_ref = []
    task_pos_ref_coeff = []
    for nn in range(n_robots.omni):
        task_pos_ref.append(RobCont(omni=ca.vertcat(s_kp1.omni[0], s_kp1.omni[1])))
        task_pos_ref_coeff.append(
            RobCont(
                omni=[[goals[nn]] for _ in range(n_robots.omni)],
            )
        )

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

    for task_obs in task_obs_avoidance:
        hompc.create_task(
            name='obstacle_avoidance',
            prio=2,
            type=TaskType.Same,
            ineq_task_ls=task_obs,
        )
    for nn in range(n_robots.omni):
        hompc.create_task(
            name=f'pos_ref_{nn}',
            prio=4,
            type=TaskType.Same,
            eq_task_ls=task_pos_ref[nn].tolist(),
            eq_task_coeff=task_pos_ref_coeff[nn].tolist(),
            # time_index=[2],
            robot_index=[[nn]],
        )

    # ======================================================================= #

    s = RobCont(omni=s_init)

    n_steps = steps

    s_history = [None for _ in range(n_steps)]
    time_goal = 0
    step_goal = 0

    last_step = n_steps
    for k in range(n_steps):
        if np.all(np.abs(np.array(s.omni)[:, :2] - goals) < 10e-3) and step_goal == 0:
            time_goal = time.time() - time_start
            step_goal = k
            print(f'scored a goal at {k}: {time_goal}')

        time_coord_start = time.time()
        time_round = time_start - time.time()

        u_star = hompc(copy.deepcopy(s.tolist()))

        s = evolve(copy.deepcopy(s), RobCont(omni=u_star[0]), dt)

        with open(filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            row = [k, time_round]

            for i in range(n_robots.omni):
                row.extend(s.omni[i])
                row.extend(u_star[0][i])

            writer.writerow(row)
        b.update(k)
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
            f'dt: {dt}\n n_c: {hompc.n_control}\n time to goal {time_goal} at {step_goal}\ntime elapsed: {time_elapsed}s\ntotal solving {time_solution}\n'
        )

    # ========================= Visualization Options ======================== #

    # robot_pairs = list(combinations(range(num_robots), 2))
    # x = np.arange(1, last_step + 1) * dt
    # # plt.figure(figsize=(10, 6))
    # for i, dist_list in enumerate(pairwise_distances):
    #     plt.plot(x, dist_list, label=f'Robots {robot_pairs[i]}')
    # plt.axhline(y=0.5, color='green', lw=4, linestyle='--')
    # plt.title('Time Evolution of Pairwise Robot Distances')
    # plt.xlabel('Time Step')
    # plt.ylabel('Distance')
    # # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig(f'{out_dir}/distances_cntr.pdf', bbox_inches='tight', format='pdf')
    # plt.close()

    visual_method = 'None'

    s_history = [s.tolist() + [[]] for s in s_history[:last_step]]

    flags = MultiRobotArtistFlags()
    flags.omnidir = False
    flags.unicycle = True
    flags.voronoi = False

    # goal = [
    #     [6, -6],
    #     [-6, -6],
    #     [-6, 6],
    #     [6, 6],
    #     [8, 3],
    #     [-8, 3],
    #     [8, -3],
    #     [-8, -3],
    #     [8, 0],
    #     [-8, 0],
    #     [0, 6],
    #     [0, -6],
    # ]

    # save_snapshots(
    #     s_history,
    #     None,
    #     [[0, 6, 4.5], [0, -6, 4.5]],
    #     dt,
    #     [(last_step - 1) * dt],
    #     f'{out_dir}/snapshot_cntr',
    #     x_lim=[-7, 7],
    #     y_lim=[-5, 5],
    #     flags=flags,
    # )

    # if visual_method is not None and visual_method != 'None':
    #     display_animation(
    #         s_history,
    #         None,
    #         [[0, 6, 4.5], [0, -6, 4.5]],
    #         dt,
    #         visual_method,
    #         x_lim=[-7.5, 7.5],
    #         y_lim=[-5, 5],
    #         video_name=f'{out_dir}/video_central.mp4',
    #         flags=flags,
    #     )

    b.finish()

    return time_elapsed


if __name__ == '__main__':
    for _ in range(1):
        n_robots = 20
        steps = 1200
        for i in range(1):
            main(n_robots, steps)
            n_robots += 10
