import copy
import time
from itertools import combinations

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist

from distributed_ho_mpc.ho_mpc.robot_models import get_unicycle_model
from hierarchical_optimization_mpc.ho_mpc_multi_robot import (
    HOMPCMultiRobot,
    QPSolver,
    TaskBiCoeff,
    TaskIndexes,
    TaskType,
)
from hierarchical_optimization_mpc.tasks_creator_ho_mpc_mr import (
    TasksCreatorHOMPCMultiRobot,
)
from hierarchical_optimization_mpc.utils.disp_het_multi_rob import (
    MultiRobotArtists,
    display_animation,
    plot_distances,
    save_snapshots,
)
from hierarchical_optimization_mpc.utils.robot_models import (
    RobCont,
    get_omnidirectional_model,
)


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

    # for j, _ in enumerate(s.omni):

    #     for _ in range(n_intervals):
    #         s.omni[j] = s.omni[j] + dt / n_intervals * np.array([
    #             u_star.omni[j][0],
    #             u_star.omni[j][1],
    #         ])

    return s


def main():
    np.random.seed(1)

    time_start = time.time()

    # ============================== Parameters ============================= #

    dt = 0.05

    n_robots = RobCont(omni=12)

    v_max = 1.8
    v_min = -1

    # ======================= Define The System Model ======================= #

    s = RobCont(omni=None)
    u = RobCont(omni=None)
    s_kp1 = RobCont(omni=None)

    s.omni, u.omni, s_kp1.omni = get_unicycle_model(dt * 10)

    # =========================== Define The Tasks ========================== #

    tasks_creator = TasksCreatorHOMPCMultiRobot(
        s.tolist(),
        u.tolist(),
        s_kp1.tolist(),
        dt,
        n_robots.tolist(),
    )

    task_input_limits = RobCont(
        omni=ca.vertcat(
            u.omni[0] - v_max,
            -u.omni[0] + 0,  # v_min,
            u.omni[1] - 1.5,  # 1v_max,
            -u.omni[1] - 1.5,  # v_min
        )
    )

    # =========================== prio 3 ======================================= #

    aux = ca.SX.sym('aux', 2, 2)
    mapping = RobCont(omni=ca.vertcat(s.omni[0], s.omni[1]))
    task_formation = ca.vertcat(
        (aux[0, 0] - aux[1, 0]) ** 2 + (aux[0, 1] - aux[1, 1]) ** 2 - 0,
    )
    task_formation_coeff = [
        TaskBiCoeff(0, 0, 0, 1, 0, 2**2),
        TaskBiCoeff(0, 2, 0, 3, 0, 2**2),
        TaskBiCoeff(0, 3, 0, 4, 0, 2**2),
    ]

    # ======================================================================= #

    task_pos_ref_1 = RobCont(omni=ca.vertcat(s_kp1.omni[0], s_kp1.omni[1]))
    task_pos_ref_1_coeff = RobCont(
        omni=[[np.array([5, -6])] for _ in range(n_robots.omni)],
    )

    # ======================================================================= #

    task_pos_ref_2 = RobCont(omni=ca.vertcat(s_kp1.omni[0], s_kp1.omni[1]))
    task_pos_ref_2_coeff = RobCont(omni=[[np.array([-5, -6])] for _ in range(n_robots.omni)])

    # ======================================================================= #

    task_pos_ref_3 = RobCont(omni=ca.vertcat(s_kp1.omni[0], s_kp1.omni[1]))
    task_pos_ref_3_coeff = RobCont(omni=[[np.array([-5, 6])] for _ in range(n_robots.omni)])

    # ======================================================================= #

    task_pos_ref_4 = RobCont(omni=ca.vertcat(s_kp1.omni[0], s_kp1.omni[1]))
    task_pos_ref_4_coeff = RobCont(omni=[[np.array([5, 6])] for _ in range(n_robots.omni)])
    # ======================================================================= #

    task_pos_ref_5 = RobCont(omni=ca.vertcat(s_kp1.omni[0], s_kp1.omni[1]))
    task_pos_ref_5_coeff = RobCont(omni=[[np.array([8, 3])] for _ in range(n_robots.omni)])

    # ======================================================================= #

    task_pos_ref_6 = RobCont(omni=ca.vertcat(s_kp1.omni[0], s_kp1.omni[1]))
    task_pos_ref_6_coeff = RobCont(omni=[[np.array([-8, 3])] for _ in range(n_robots.omni)])
    # ======================================================================= #

    task_pos_ref_7 = RobCont(omni=ca.vertcat(s_kp1.omni[0], s_kp1.omni[1]))
    task_pos_ref_7_coeff = RobCont(omni=[[np.array([8, -3])] for _ in range(n_robots.omni)])

    # ======================================================================= #

    task_pos_ref_8 = RobCont(omni=ca.vertcat(s_kp1.omni[0], s_kp1.omni[1]))
    task_pos_ref_8_coeff = RobCont(omni=[[np.array([-8, -3])] for _ in range(n_robots.omni)])
    # ======================================================================= #
    task_pos_ref_9 = RobCont(omni=ca.vertcat(s_kp1.omni[0], s_kp1.omni[1]))
    task_pos_ref_9_coeff = RobCont(omni=[[np.array([9, 0])] for _ in range(n_robots.omni)])

    # ======================================================================= #

    task_pos_ref_10 = RobCont(omni=ca.vertcat(s_kp1.omni[0], s_kp1.omni[1]))
    task_pos_ref_10_coeff = RobCont(omni=[[np.array([-9, 0])] for _ in range(n_robots.omni)])
    # ======================================================================= #

    task_pos_ref_11 = RobCont(omni=ca.vertcat(s_kp1.omni[0], s_kp1.omni[1]))
    task_pos_ref_11_coeff = RobCont(omni=[[np.array([0, 6])] for _ in range(n_robots.omni)])

    # ======================================================================= #

    task_pos_ref_12 = RobCont(omni=ca.vertcat(s_kp1.omni[0], s_kp1.omni[1]))
    task_pos_ref_12_coeff = RobCont(omni=[[np.array([0, -6])] for _ in range(n_robots.omni)])
    # ======================================================================= #

    task_input_min = RobCont(omni=ca.vertcat(u.omni[0], u.omni[1]))

    # ======================================================================= #

    threshold = 2
    aux_avoid_collision = ca.SX.sym('aux', 2, 2)
    mapping_avoid_collision = RobCont(omni=ca.vertcat(s.omni[0], s.omni[1]))
    task_avoid_collision = ca.vertcat(
        -((aux_avoid_collision[0, 0] - aux_avoid_collision[1, 0]) ** 2)
        - (aux_avoid_collision[0, 1] - aux_avoid_collision[1, 1]) ** 2,
    )
    task_avoid_collision_coeff = [
        TaskBiCoeff(0, 0, 0, 1, 0, -(threshold**2)),
        TaskBiCoeff(0, 0, 0, 2, 0, -(threshold**2)),
        TaskBiCoeff(0, 0, 0, 3, 0, -(threshold**2)),
        TaskBiCoeff(0, 0, 0, 4, 0, -(threshold**2)),
        TaskBiCoeff(0, 0, 0, 5, 0, -(threshold**2)),
        TaskBiCoeff(0, 0, 0, 6, 0, -(threshold**2)),
        TaskBiCoeff(0, 0, 0, 7, 0, -(threshold**2)),
        TaskBiCoeff(0, 0, 0, 8, 0, -(threshold**2)),
        TaskBiCoeff(0, 0, 0, 9, 0, -(threshold**2)),
        TaskBiCoeff(0, 0, 0, 10, 0, -(threshold**2)),
        TaskBiCoeff(0, 0, 0, 11, 0, -(threshold**2)),
        TaskBiCoeff(0, 1, 0, 2, 0, -(threshold**2)),
        TaskBiCoeff(0, 1, 0, 3, 0, -(threshold**2)),
        TaskBiCoeff(0, 1, 0, 4, 0, -(threshold**2)),
        TaskBiCoeff(0, 1, 0, 5, 0, -(threshold**2)),
        TaskBiCoeff(0, 1, 0, 6, 0, -(threshold**2)),
        TaskBiCoeff(0, 1, 0, 7, 0, -(threshold**2)),
        TaskBiCoeff(0, 1, 0, 8, 0, -(threshold**2)),
        TaskBiCoeff(0, 1, 0, 9, 0, -(threshold**2)),
        TaskBiCoeff(0, 1, 0, 10, 0, -(threshold**2)),
        TaskBiCoeff(0, 1, 0, 11, 0, -(threshold**2)),
        TaskBiCoeff(0, 2, 0, 3, 0, -(threshold**2)),
        TaskBiCoeff(0, 2, 0, 4, 0, -(threshold**2)),
        TaskBiCoeff(0, 2, 0, 5, 0, -(threshold**2)),
        TaskBiCoeff(0, 2, 0, 6, 0, -(threshold**2)),
        TaskBiCoeff(0, 2, 0, 7, 0, -(threshold**2)),
        TaskBiCoeff(0, 2, 0, 8, 0, -(threshold**2)),
        TaskBiCoeff(0, 2, 0, 9, 0, -(threshold**2)),
        TaskBiCoeff(0, 2, 0, 10, 0, -(threshold**2)),
        TaskBiCoeff(0, 2, 0, 11, 0, -(threshold**2)),
        TaskBiCoeff(0, 3, 0, 4, 0, -(threshold**2)),
        TaskBiCoeff(0, 3, 0, 5, 0, -(threshold**2)),
        TaskBiCoeff(0, 3, 0, 6, 0, -(threshold**2)),
        TaskBiCoeff(0, 3, 0, 7, 0, -(threshold**2)),
        TaskBiCoeff(0, 3, 0, 8, 0, -(threshold**2)),
        TaskBiCoeff(0, 3, 0, 9, 0, -(threshold**2)),
        TaskBiCoeff(0, 3, 0, 10, 0, -(threshold**2)),
        TaskBiCoeff(0, 3, 0, 11, 0, -(threshold**2)),
        TaskBiCoeff(0, 4, 0, 5, 0, -(threshold**2)),
        TaskBiCoeff(0, 4, 0, 6, 0, -(threshold**2)),
        TaskBiCoeff(0, 4, 0, 7, 0, -(threshold**2)),
        TaskBiCoeff(0, 4, 0, 8, 0, -(threshold**2)),
        TaskBiCoeff(0, 4, 0, 9, 0, -(threshold**2)),
        TaskBiCoeff(0, 4, 0, 10, 0, -(threshold**2)),
        TaskBiCoeff(0, 4, 0, 11, 0, -(threshold**2)),
        TaskBiCoeff(0, 5, 0, 6, 0, -(threshold**2)),
        TaskBiCoeff(0, 5, 0, 7, 0, -(threshold**2)),
        TaskBiCoeff(0, 5, 0, 8, 0, -(threshold**2)),
        TaskBiCoeff(0, 5, 0, 9, 0, -(threshold**2)),
        TaskBiCoeff(0, 5, 0, 10, 0, -(threshold**2)),
        TaskBiCoeff(0, 5, 0, 11, 0, -(threshold**2)),
        TaskBiCoeff(0, 6, 0, 7, 0, -(threshold**2)),
        TaskBiCoeff(0, 6, 0, 8, 0, -(threshold**2)),
        TaskBiCoeff(0, 6, 0, 9, 0, -(threshold**2)),
        TaskBiCoeff(0, 6, 0, 10, 0, -(threshold**2)),
        TaskBiCoeff(0, 6, 0, 11, 0, -(threshold**2)),
        TaskBiCoeff(0, 7, 0, 8, 0, -(threshold**2)),
        TaskBiCoeff(0, 7, 0, 9, 0, -(threshold**2)),
        TaskBiCoeff(0, 7, 0, 10, 0, -(threshold**2)),
        TaskBiCoeff(0, 7, 0, 11, 0, -(threshold**2)),
        TaskBiCoeff(0, 8, 0, 9, 0, -(threshold**2)),
        TaskBiCoeff(0, 8, 0, 10, 0, -(threshold**2)),
        TaskBiCoeff(0, 8, 0, 11, 0, -(threshold**2)),
        TaskBiCoeff(0, 9, 0, 10, 0, -(threshold**2)),
        TaskBiCoeff(0, 9, 0, 11, 0, -(threshold**2)),
        TaskBiCoeff(0, 10, 0, 11, 0, -(threshold**2)),
    ]

    # aux_avoid_collision, mapping_avoid_collision, task_avoid_collision, task_avoid_collision_coeff = tasks_creator.get_task_avoid_collision(0.5)

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
    hompc.create_task(
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

    hompc.create_task_bi(
        name='collision_avoidance',
        prio=3,
        type=TaskType.Bi,
        aux=aux_avoid_collision,
        mapping=mapping_avoid_collision.tolist(),
        ineq_task_ls=task_avoid_collision,
        ineq_task_coeff=task_avoid_collision_coeff,
    )
    # hompc.create_task_bi(
    #    name="formation", prio=3,
    #    type=TaskType.Bi,
    #    aux=aux,
    #    mapping=mapping.tolist(),
    #    eq_task_ls=task_formation,
    #    eq_task_coeff=task_formation_coeff,
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

    # s = RobCont(omni=
    #     [np.multiply(np.random.random((2)), np.array([2, 2])) + np.array([-1, -1])
    #      for _ in range(n_robots.omni)],
    # )
    s = RobCont(
        omni=[
            np.array([-5, 5, 0.1]),
            np.array([4.5, 5, -2.1]),
            np.array([4.5, -5, 2.1]),
            np.array([-5, -5, 0.75]),
            np.array([-6, -3, 0.25]),
            np.array([6, 3, 3]),
            np.array([-6, 3, 0.25]),
            np.array([6, -3, 3]),
            np.array([-7, 0, 0.25]),
            np.array([7, 0, 3]),
            np.array([0, -5, 2.1]),
            np.array([0, 5, -2.1]),
        ]
    )

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

    n_steps = 1500

    s_history = [None for _ in range(n_steps)]

    goals = np.array(
        [
            [5, -6],
            [-5, -6],
            [-5, 6],
            [5, 6],
            [8, 3],
            [-8, 3],
            [8, -3],
            [-8, -3],
            [9, 0],
            [-9, 0],
            [0, 6],
            [0, -6],
        ]
    )

    for k in range(n_steps):
        if np.all(np.abs(np.array(s.omni)[:, :2] - goals) < 10e-3):
            break

        time_coord_start = time.time()
        print(k)

        u_star = hompc(copy.deepcopy(s.tolist()))

        print(f's: {s}')
        print(f'u_star: {u_star}')
        print()

        s = evolve(s, RobCont(omni=u_star[0]), dt)

        s_history[k] = copy.deepcopy(s)
        pairwise_distances = agents_distance(s.tolist()[0], pairwise_distances)

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
    x = np.arange(1, n_steps + 1) * dt
    plt.figure(figsize=(10, 6))
    for i, dist_list in enumerate(pairwise_distances):
        plt.plot(x, dist_list, label=f'Robots {robot_pairs[i]}')

    plt.title('Time Evolution of Pairwise Robot Distances')
    plt.xlabel('Time Step')
    plt.ylabel('Distance')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    visual_method = 'plot'

    s_history = [[[]] + s.tolist() for s in s_history]

    artist_flags = MultiRobotArtists(
        centroid=False,
        goals=True,
        obstacles=False,
        past_trajectory=False,
        omnidir=RobCont(omni=True),
        unicycles=False,
        # robots=RobCont(omni=True),
        # robot_names=True,
        voronoi=False,
    )
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
    if visual_method is not None and visual_method != 'none':
        display_animation(
            s_history,
            goal,
            None,
            dt,
            visual_method,
            artist_flags,
        )

    if visual_method == 'save':
        save_snapshots(
            s_history,
            None,
            dt,
            [0, 10, 25],
            'snapshot',
            artist_flags,
        )

    return time_elapsed


if __name__ == '__main__':
    main()
