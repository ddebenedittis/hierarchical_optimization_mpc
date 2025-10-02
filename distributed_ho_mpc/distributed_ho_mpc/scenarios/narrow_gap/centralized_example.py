import copy
import csv
import os
import time
from datetime import datetime
from itertools import combinations

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
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


def main():
    np.random.seed(1)

    package_name = 'distributed_ho_mpc'
    workspace_dir = f'{get_package_share_directory(package_name)}/../../../..'
    out_dir = f'{workspace_dir}/out/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}-narrow_gap/'
    os.makedirs(out_dir, exist_ok=True)

    time_start = time.time()

    # ============================== Parameters ============================= #

    dt = 0.08

    n_robots = RobCont(omni=4)

    v_max = 1.5
    v_min = -1

    # ======================= Define The System Model ======================= #

    s = RobCont(omni=None)
    u = RobCont(omni=None)
    s_kp1 = RobCont(omni=None)

    s.omni, u.omni, s_kp1.omni = get_unicycle_model(dt * 10)

    # ========================== Prepare The Data ========================== #
    filename = f'{out_dir}/cntr_data.csv'
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)

        header = ['Time']
        for i in range(n_robots.omni):
            header.append(f'stateX_{i}')
            header.append(f'stateY_{i}')
            header.append(f'stateTheta_{i}')
            header.append(f'inputV_{i}')
            header.append(f'inputOH_{i}')
        for i in range(5):
            header.append(f'cost_p{i}')

        writer.writerow(header)

    # =========================== Define The Tasks ========================== #

    task_input_limits = RobCont(
        omni=ca.vertcat(
            u.omni[0] - v_max,
            -u.omni[0] + v_min,  # v_min,
            u.omni[1] - 1.4,  # 1v_max,
            -u.omni[1] - 1.4,  # v_min
        )
    )

    # ===============================
    obstacle_pos_1 = np.array([0, 6])
    obstacle_pos_2 = np.array([0, -6])
    obstacle_size = 4.5
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

    task_pos_ref_1 = RobCont(omni=ca.vertcat(s_kp1.omni[0], s_kp1.omni[1]))
    task_pos_ref_1_coeff = RobCont(
        omni=[[np.array([3, 2])] for _ in range(n_robots.omni)],
    )

    # ======================================================================= #

    task_pos_ref_2 = RobCont(omni=ca.vertcat(s_kp1.omni[0], s_kp1.omni[1]))
    task_pos_ref_2_coeff = RobCont(omni=[[np.array([-3, 2])] for _ in range(n_robots.omni)])

    # ======================================================================= #

    task_pos_ref_3 = RobCont(omni=ca.vertcat(s_kp1.omni[0], s_kp1.omni[1]))
    task_pos_ref_3_coeff = RobCont(omni=[[np.array([3, -2])] for _ in range(n_robots.omni)])

    # ======================================================================= #

    task_pos_ref_4 = RobCont(omni=ca.vertcat(s_kp1.omni[0], s_kp1.omni[1]))
    task_pos_ref_4_coeff = RobCont(omni=[[np.array([-3, -2])] for _ in range(n_robots.omni)])
    # ======================================================================= #

    task_pos_ref_5 = RobCont(omni=ca.vertcat(s_kp1.omni[0], s_kp1.omni[1]))
    task_pos_ref_5_coeff = RobCont(omni=[[np.array([4, -1])] for _ in range(n_robots.omni)])

    # ======================================================================= #

    task_pos_ref_6 = RobCont(omni=ca.vertcat(s_kp1.omni[0], s_kp1.omni[1]))
    task_pos_ref_6_coeff = RobCont(omni=[[np.array([-4, -1])] for _ in range(n_robots.omni)])
    # ======================================================================= #

    task_pos_ref_7 = RobCont(omni=ca.vertcat(s_kp1.omni[0], s_kp1.omni[1]))
    task_pos_ref_7_coeff = RobCont(omni=[[np.array([4, 1])] for _ in range(n_robots.omni)])

    # ======================================================================= #

    task_pos_ref_8 = RobCont(omni=ca.vertcat(s_kp1.omni[0], s_kp1.omni[1]))
    task_pos_ref_8_coeff = RobCont(omni=[[np.array([-4, 1])] for _ in range(n_robots.omni)])
    # ======================================================================= #
    task_pos_ref_9 = RobCont(omni=ca.vertcat(s_kp1.omni[0], s_kp1.omni[1]))
    task_pos_ref_9_coeff = RobCont(omni=[[np.array([5, 0])] for _ in range(n_robots.omni)])

    # ======================================================================= #

    task_pos_ref_10 = RobCont(omni=ca.vertcat(s_kp1.omni[0], s_kp1.omni[1]))
    task_pos_ref_10_coeff = RobCont(omni=[[np.array([-5, 0])] for _ in range(n_robots.omni)])
    # ======================================================================= #
    task_pos_ref_11 = RobCont(omni=ca.vertcat(s_kp1.omni[0], s_kp1.omni[1]))
    task_pos_ref_11_coeff = RobCont(omni=[[np.array([-4, 0])] for _ in range(n_robots.omni)])
    # ======================================================================= #
    task_pos_ref_12 = RobCont(omni=ca.vertcat(s_kp1.omni[0], s_kp1.omni[1]))
    task_pos_ref_12_coeff = RobCont(omni=[[np.array([4, 0])] for _ in range(n_robots.omni)])
    # ======================================================================= #
    task_pos_ref_13 = RobCont(omni=ca.vertcat(s_kp1.omni[0], s_kp1.omni[1]))
    task_pos_ref_13_coeff = RobCont(omni=[[np.array([6, 1])] for _ in range(n_robots.omni)])

    # ======================================================================= #

    task_pos_ref_14 = RobCont(omni=ca.vertcat(s_kp1.omni[0], s_kp1.omni[1]))
    task_pos_ref_14_coeff = RobCont(omni=[[np.array([-6, 1])] for _ in range(n_robots.omni)])
    # ======================================================================= #
    task_pos_ref_15 = RobCont(omni=ca.vertcat(s_kp1.omni[0], s_kp1.omni[1]))
    task_pos_ref_15_coeff = RobCont(omni=[[np.array([6, -1])] for _ in range(n_robots.omni)])
    # ======================================================================= #
    task_pos_ref_16 = RobCont(omni=ca.vertcat(s_kp1.omni[0], s_kp1.omni[1]))
    task_pos_ref_16_coeff = RobCont(omni=[[np.array([-6, -1])] for _ in range(n_robots.omni)])
    # ======================================================================= #
    task_pos_ref_17 = RobCont(omni=ca.vertcat(s_kp1.omni[0], s_kp1.omni[1]))
    task_pos_ref_17_coeff = RobCont(omni=[[np.array([7, -2])] for _ in range(n_robots.omni)])
    # ======================================================================= #
    task_pos_ref_18 = RobCont(omni=ca.vertcat(s_kp1.omni[0], s_kp1.omni[1]))
    task_pos_ref_18_coeff = RobCont(omni=[[np.array([-7, -2])] for _ in range(n_robots.omni)])

    threshold = 0.5
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
    hompc.n_control = 1
    hompc.n_pred = 0

    hompc.create_task(
        name='input_limits',
        prio=1,
        type=TaskType.Same,
        ineq_task_ls=task_input_limits.tolist(),
    )
    hompc.create_task(
        name='pos_ref_1',
        prio=4,
        type=TaskType.Same,
        eq_task_ls=task_pos_ref_1.tolist(),
        eq_task_coeff=task_pos_ref_1_coeff.tolist(),
        robot_index=[[0]],
    )
    hompc.create_task(
        name='pos_ref_2',
        prio=4,
        type=TaskType.Same,
        eq_task_ls=task_pos_ref_2.tolist(),
        eq_task_coeff=task_pos_ref_2_coeff.tolist(),
        robot_index=[[1]],
    )
    hompc.create_task(
        name='pos_ref_3',
        prio=4,
        type=TaskType.Same,
        eq_task_ls=task_pos_ref_3.tolist(),
        eq_task_coeff=task_pos_ref_3_coeff.tolist(),
        robot_index=[[2]],
    )
    hompc.create_task(
        name='pos_ref_4',
        prio=4,
        type=TaskType.Same,
        eq_task_ls=task_pos_ref_4.tolist(),
        eq_task_coeff=task_pos_ref_4_coeff.tolist(),
        robot_index=[[3]],
    )
    """hompc.create_task(
        name='pos_ref_5',
        prio=4,
        type=TaskType.Same,
        eq_task_ls=task_pos_ref_5.tolist(),
        eq_task_coeff=task_pos_ref_5_coeff.tolist(),
        robot_index=[[4]],
    )
    hompc.create_task(
        name='pos_ref_6',
        prio=4,
        type=TaskType.Same,
        eq_task_ls=task_pos_ref_6.tolist(),
        eq_task_coeff=task_pos_ref_6_coeff.tolist(),
        robot_index=[[5]],
    )
    hompc.create_task(
        name='pos_ref_7',
        prio=4,
        type=TaskType.Same,
        eq_task_ls=task_pos_ref_7.tolist(),
        eq_task_coeff=task_pos_ref_7_coeff.tolist(),
        robot_index=[[6]],
    )
    hompc.create_task(
        name='pos_ref_8',
        prio=4,
        type=TaskType.Same,
        eq_task_ls=task_pos_ref_8.tolist(),
        eq_task_coeff=task_pos_ref_8_coeff.tolist(),
        robot_index=[[7]],
    )
    hompc.create_task(
        name='pos_ref_9',
        prio=4,
        type=TaskType.Same,
        eq_task_ls=task_pos_ref_9.tolist(),
        eq_task_coeff=task_pos_ref_9_coeff.tolist(),
        robot_index=[[8]],
    )
    hompc.create_task(
        name='pos_ref_10',
        prio=4,
        type=TaskType.Same,
        eq_task_ls=task_pos_ref_10.tolist(),
        eq_task_coeff=task_pos_ref_10_coeff.tolist(),
        robot_index=[[9]],
    )
    
    hompc.create_task(
        name='pos_ref_11',
        prio=4,
        type=TaskType.Same,
        eq_task_ls=task_pos_ref_11.tolist(),
        eq_task_coeff=task_pos_ref_11_coeff.tolist(),
        robot_index=[[10]],
    )
    hompc.create_task(
        name='pos_ref_12',
        prio=4,
        type=TaskType.Same,
        eq_task_ls=task_pos_ref_12.tolist(),
        eq_task_coeff=task_pos_ref_12_coeff.tolist(),
        robot_index=[[11]],
    )
    hompc.create_task(
        name='pos_ref_13',
        prio=4,
        type=TaskType.Same,
        eq_task_ls=task_pos_ref_13.tolist(),
        eq_task_coeff=task_pos_ref_13_coeff.tolist(),
        robot_index=[[12]],
    )
    hompc.create_task(
        name='pos_ref_14',
        prio=4,
        type=TaskType.Same,
        eq_task_ls=task_pos_ref_14.tolist(),
        eq_task_coeff=task_pos_ref_14_coeff.tolist(),
        robot_index=[[13]],
    )
    hompc.create_task(
        name='pos_ref_15',
        prio=4,
        type=TaskType.Same,
        eq_task_ls=task_pos_ref_15.tolist(),
        eq_task_coeff=task_pos_ref_15_coeff.tolist(),
        robot_index=[[14]],
    )
    hompc.create_task(
        name='pos_ref_16',
        prio=4,
        type=TaskType.Same,
        eq_task_ls=task_pos_ref_16.tolist(),
        eq_task_coeff=task_pos_ref_16_coeff.tolist(),
        robot_index=[[15]],
    )
    hompc.create_task(
        name='pos_ref_17',
        prio=4,
        type=TaskType.Same,
        eq_task_ls=task_pos_ref_17.tolist(),
        eq_task_coeff=task_pos_ref_17_coeff.tolist(),
        robot_index=[[16]],
    )
    hompc.create_task(
        name='pos_ref_18',
        prio=4,
        type=TaskType.Same,
        eq_task_ls=task_pos_ref_18.tolist(),
        eq_task_coeff=task_pos_ref_18_coeff.tolist(),
        robot_index=[[17]],
    )"""

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

    # ======================================================================= #

    s = RobCont(
        omni=[
            np.array([-5, 2, -0.3]),
            np.array([5, 2, -3]),
            np.array([-5, -2, 0.1]),
            np.array([5, -2, 2.8]),
        ]
    )
    # np.array([-6.5, -1, 0]),
    # np.array([6.5, -1, -3]),
    # np.array([-6.5, 1, -0.1]),
    # np.array([6.5, 1, 3.14]),
    # np.array([-5, 0, 0]),
    # np.array([5, 0, 3.14]),
    # np.array([7, 0, 3.14]),
    # np.array([-7, 0, 0.0]),
    # np.array([-7.5, 1, -0.04]),
    # np.array([7.5, 1, -3.11]),
    # np.array([-7.5, -1, 0.04]),
    # np.array([7.5, -1, 3.11]),
    # np.array([-8, 0, 0.04]),
    # np.array([8, 0, -3.11]),

    def agents_distance(state, pairwise_distances):
        """
        Plot the distance between the agents at each time step
        """
        positions_over_time = np.array(state)
        distances = pdist(positions_over_time, metric='euclidean')  # shape: (num_pairs,)
        for i, d in enumerate(distances):
            pairwise_distances[i].append(d)
        return pairwise_distances

    num_robots = n_robots.omni
    num_pairs = int(num_robots * (num_robots - 1) / 2)

    # Initialize one list per robot pair
    pairwise_distances = [[] for _ in range(num_pairs)]

    n_steps = 3500

    s_history = [None for _ in range(n_steps)]

    goals = np.array(
        [
            [3, 2],
            [-3, 2],
            [3, -2],
            [-3, -2],
            # [4, -1],
            # [-4, -1],
            # [4, 1],
            # [-4, 1],
            # [5, 0],
            # [-5, 0],
            # [-4, 0],
            # [4, 0],
            # [6, 1],
            # [-6, 1],
            # [6, -1],
            # [-6, -1],
            # [7, -2],
            # [-7, -2],
        ]
    )
    last_step = n_steps
    for k in range(n_steps):
        if np.all(np.abs(np.array(s.omni)[:, :2] - goals) < 10e-3):
            last_step = k
            break

        time_coord_start = time.time()
        print(k)

        u_star, cost = hompc(copy.deepcopy(s.tolist()))

        print(f's: {s}')
        print(f'u_star: {u_star}')
        print()

        s = evolve(s, RobCont(omni=u_star[0]), dt)
        for theta in s.omni:
            theta[2] = (theta[2] % (2 * np.pi) + 2 * np.pi) % (2 * np.pi)

        with open(filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            row = [k]

            for i in range(n_robots.omni):
                row.extend(s.omni[i])
                row.extend(u_star[0][i])
            row.extend(cost.tolist())

            writer.writerow(row)

        s_history[k] = copy.deepcopy(s)
        pairwise_distances = agents_distance(s.tolist()[0], pairwise_distances)
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

    # ========================= Visualization Options ======================== #

    robot_pairs = list(combinations(range(num_robots), 2))
    x = np.arange(1, last_step + 1) * dt
    # plt.figure(figsize=(10, 6))
    for i, dist_list in enumerate(pairwise_distances):
        plt.plot(x, dist_list, label=f'Robots {robot_pairs[i]}')
    plt.axhline(y=0.5, color='green', lw=4, linestyle='--')
    plt.title('Time Evolution of Pairwise Robot Distances')
    plt.xlabel('Time Step')
    plt.ylabel('Distance')
    # plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/distances_cntr.pdf', bbox_inches='tight', format='pdf')
    plt.close()

    visual_method = 'None'

    s_history = [s.tolist() + [[]] for s in s_history[:last_step]]

    flags = MultiRobotArtistFlags()
    flags.omnidir = False
    flags.unicycle = True
    flags.voronoi = False

    goal = [
        [6, -6],
        [-6, -6],
        [-6, 6],
        [6, 6],
        [8, 3],
        [-8, 3],
        [8, -3],
        [-8, -3],
        [8, 0],
        [-8, 0],
        [0, 6],
        [0, -6],
    ]

    save_snapshots(
        s_history,
        None,
        [[0, 6, 4.5], [0, -6, 4.5]],
        dt,
        [(last_step - 1) * dt],
        f'{out_dir}/snapshot_cntr',
        x_lim=[-7, 7],
        y_lim=[-5, 5],
        flags=flags,
    )

    if visual_method is not None and visual_method != 'None':
        display_animation(
            s_history,
            None,
            [[0, 6, 4.5], [0, -6, 4.5]],
            dt,
            visual_method,
            x_lim=[-7.5, 7.5],
            y_lim=[-5, 5],
            video_name=f'{out_dir}/video_central.mp4',
            flags=flags,
        )

    print(f'The time elapsed is {time_elapsed} seconds')
    print(f'The time elapsed for coordination is {time_coord} seconds')

    print('The time was used in the following phases:')
    max_key_len = max(map(len, hompc.solve_times.keys()))
    for key, value in hompc.solve_times.items():
        key_len = len(key)
        print(f'{key}: {" " * (max_key_len - key_len)}{value}')

    return time_elapsed


if __name__ == '__main__':
    main()
