import copy
import os
import time
from datetime import datetime
from itertools import combinations

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import progressbar
from ament_index_python.packages import get_package_share_directory
from scipy.spatial.distance import pdist

import distributed_ho_mpc.scenarios.radial_switching_unicycle.settings as st
from distributed_ho_mpc.scenarios.radial_switching_unicycle.node import Node
from hierarchical_optimization_mpc.utils.disp_het_multi_rob import (
    MultiRobotArtistFlags,
    display_animation,
    plot_distances,
    save_snapshots,
)
from hierarchical_optimization_mpc.utils.robot_models import (
    get_omnidirectional_model,
    get_unicycle_model,
)


def main(n_robot):
    np.random.seed(1)
    b = progressbar.ProgressBar(maxval=st.n_steps)
    b.start()

    model = {
        'unicycle': get_unicycle_model(st.dt),
        'omnidirectional': get_omnidirectional_model(st.dt),
    }

    def neigh_connection(states, nodes, graph_matrix, communication_range):
        """
        For each node, connect to up to 6 nearest neighbors within communication range.
        Disconnect from neighbors outside range or beyond top 6 closest.
        """
        num_nodes = len(nodes)

        for i in range(num_nodes):
            distances = []

            for j in range(num_nodes):
                if i == j:
                    continue
                dist = np.linalg.norm(states[i][:2] - states[j][:2])
                if dist < communication_range:
                    distances.append((j, dist))

            # Sort and select up to 6 nearest within range
            distances.sort(key=lambda x: x[1])
            if len(distances) > 0:
                if distances[0][1] < 4:
                    nodes[i].a = 1
                elif distances[0][1] > 4:
                    nodes[i].a = 5
            closest_neighbors = set(idx for idx, _ in distances[: st.limit_connection])

            current_connections = set(np.nonzero(graph_matrix[i])[0])

            to_connect = closest_neighbors - current_connections
            to_disconnect = current_connections - closest_neighbors

            # --- CONNECT (bidirectional)
            for idx in to_connect:
                graph_matrix[i][idx] = 1.0
                graph_matrix[idx][i] = 1.0  # mirror connection

                # i connects to idx
                tasks_i = {
                    f'agent_{i}': {f'agent_{idx}': copy.deepcopy(system_tasks[f'agent_{idx}'])}
                }
                nodes[i].create_connection(graph_matrix[i], tasks_i[f'agent_{i}'], states[idx])

                # idx connects to i
                tasks_j = {
                    f'agent_{idx}': {f'agent_{i}': copy.deepcopy(system_tasks[f'agent_{i}'])}
                }
                nodes[idx].create_connection(graph_matrix[idx], tasks_j[f'agent_{idx}'], states[i])

            # --- DISCONNECT (bidirectional)
            for idx in to_disconnect:
                graph_matrix[i][idx] = 0.0
                graph_matrix[idx][i] = 0.0  # mirror disconnection

                # i disconnects from idx
                nodes[i].remove_connection(graph_matrix[i], f'agent_{idx}', idx)

                # idx disconnects from i
                nodes[idx].remove_connection(graph_matrix[idx], f'agent_{i}', i)

    def agents_distance(state, pairwise_distances):
        """
        Plot the distance between the agents at each time step
        """
        positions_over_time = np.array(state)
        positions_over_time = positions_over_time[:, :2]
        distances = pdist(positions_over_time, metric='euclidean')  # shape: (num_pairs,)
        for i, d in enumerate(distances):
            pairwise_distances[i].append(d)
        return pairwise_distances

    # =========================================================================== #
    #                                TASK SCHEDULER                               #
    # =========================================================================== #

    # goals = [
    #         np.array([4, 6]),
    #         np.array([-6, -8]),
    #         np.array([0,0])
    #     ]
    snap = [0]  # time for snapshot
    for tt in snap:
        if tt > st.n_steps * st.dt:
            raise ValueError('Time instant for snapshot out of simulation lenght')

    center = np.array([0, 0])  # choose the center
    num_points = n_robot

    # radius from straight-line (chord) distance = 0.8
    radius = 3 / (2 * np.sin(np.pi / num_points))

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
        if st.type == 'uni':
            s_init.append(np.array([x, y, theta + np.pi]))
        if st.type == 'omni':
            s_init.append(np.array([x, y]))

    system_tasks = {}

    for ag in range(num_points):
        system_tasks[f'agent_{ag}'] = [
            {'prio': 1, 'name': 'input_limits'},
            {'prio': 2, 'name': 'input_smooth'},
            {'prio': 3, 'name': 'collision_avoidance'},
            {'prio': 4, 'name': 'position', 'goal': goals[ag], 'goal_index': ag},
        ]

    # system_tasks = {
    #     'agent_0': [
    #         {'prio': 1, 'name': 'input_limits'},
    #         {'prio': 2, 'name': 'input_smooth'},
    #         {'prio': 3, 'name': 'collision_avoidance'},
    #         {'prio': 4, 'name': 'position', 'goal': goals[0], 'goal_index': 0},
    #     ],
    #     'agent_1': [
    #         {'prio': 1, 'name': 'input_limits'},
    #         {'prio': 2, 'name': 'input_smooth'},
    #         {'prio': 4, 'name': 'position', 'goal': goals[1], 'goal_index': 1},
    #         {'prio': 3, 'name': 'collision_avoidance'},
    #     ],
    #     'agent_2': [
    #         {'prio': 1, 'name': 'input_limits'},
    #         {'prio': 2, 'name': 'input_smooth'},
    #         {'prio': 3, 'name': 'collision_avoidance'},
    #         {'prio': 4, 'name': 'position', 'goal': goals[2], 'goal_index': 2},
    #     ],
    #     'agent_3': [
    #         {'prio': 1, 'name': 'input_limits'},
    #         {'prio': 2, 'name': 'input_smooth'},
    #         {'prio': 3, 'name': 'collision_avoidance'},
    #         {'prio': 4, 'name': 'position', 'goal': goals[3], 'goal_index': 3},
    #     ],
    # }

    # ---------------------------------------------------------------------------- #
    #               Create the network and connection between agents               #
    # ---------------------------------------------------------------------------- #

    # deterministic graphs
    if st.n_nodes == 2:
        graph_matrix = np.array([[0.0, 1.0], [1.0, 0.0]])
        network_graph = nx.from_numpy_array(graph_matrix, nodelist=[0, 1])
    if st.n_nodes == 3:
        graph_matrix = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
        network_graph = nx.from_numpy_array(graph_matrix, nodelist=[0, 1, 2])
    if st.n_nodes == 4:
        graph_matrix = np.array(
            [
                [0.0, 1.0, 1.0, 1.0],
                [1.0, 0.0, 1.0, 1.0],
                [1.0, 1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0, 0.0],
            ]
        )
        network_graph = nx.from_numpy_array(graph_matrix, nodelist=[0, 1, 2, 3])
    if st.n_nodes == 5:
        graph_matrix = np.array(
            [
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
            ]
        )
        network_graph = nx.from_numpy_array(graph_matrix, nodelist=[0, 1, 2, 3, 4])
    graph_matrix = np.zeros((n_robot, n_robot))

    # random graph 🎲
    while st.random_graph:
        network_graph = nx.gnp_random_graph(st.n_nodes, st.p)
        graph_matrix = nx.to_numpy_array(network_graph)

        test = np.linalg.matrix_power((st.I_NN + graph_matrix), st.n_nodes)

        if np.all(test > 0):
            print('the graph is connected')
            # nx.draw(network_graph)
            # plt.show()
            break
        else:
            print('the graph is NOT connected')

    # update task manifold with the neighbours tasks
    neigh_tasks = {}
    for i in range(n_robot):
        id = 0
        neigh_tasks[f'agent_{i}'] = {}
        for j in graph_matrix[i]:
            if int(j) != 0:
                neigh_tasks[f'agent_{i}'][f'agent_{id}'] = copy.deepcopy(
                    system_tasks[f'agent_{id}']
                )
            id += 1

    # ----------------------------------------------------------------------------- #
    #         Create agents and initialize them based on settings and tasks        #
    # ---------------------------------------------------------------------------- #

    nodes = []  # list of agents of the system

    package_name = 'distributed_ho_mpc'
    workspace_dir = f'{get_package_share_directory(package_name)}/../../../..'
    out_dir = f'{workspace_dir}/out/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}-radial_switching_{n_robot}A/'
    os.makedirs(out_dir, exist_ok=True)
    time_start = time.time()
    # Create an agents of the same type for each node of the system
    for i in range(n_robot):
        node = Node(
            i,  # ID
            graph_matrix[i],  # Neighbours
            model['unicycle'],  # robot model
            st.dt,  # time step
            system_tasks[f'agent_{i}'],  # agent's tasks
            neigh_tasks[f'agent_{i}'],  # neighbours tasks
            goals,  # goals to be reached
            st.n_steps,  # max simulation steps
            out_dir=out_dir,
            init_s=s_init[i],
        )
        nodes.append(node)

        # create frameworks for the agents
        nodes[i].Tasks()
        nodes[i].MPC()

    # ---------------------------------------------------------------------------- #
    #     iterate through the nodes, transmitting datas and the receiving them     #
    # ---------------------------------------------------------------------------- #

    # DISTANCES BETWEEN AGENTS
    state = [None] * n_robot  # list of x for inizialization of optimization

    num_robots = n_robot
    num_pairs = int(num_robots * (num_robots - 1) / 2)

    # Initialize one list per robot pair
    pairwise_distances = [[] for _ in range(num_pairs)]

    gg = np.array(goals[:n_robot])

    flags = MultiRobotArtistFlags()
    flags.voronoi = False
    flags.future_trajectory = False
    time_goal = 0
    start_time_coop = time.time()
    step_goal = 0
    for j in range(n_robot):
        state[j] = nodes[j].s.omni[0]  # TODO manage heterogeneous robots
    for i in range(st.n_steps):
        if i == st.n_steps - 1:
            last_step = i + 1
        if i > 0:
            neigh_connection(state, nodes, graph_matrix, st.communication_range)
        # for rr in range(st.inner_loop):
        for j in range(n_robot):
            nodes[j].reorder_s_init(state)
            nodes[j].update('2')  # Update primal solution and state evolution
        for j in range(n_robot):
            state[j] = nodes[j].s.omni[0]  # TODO manage heterogeneous robots
            # for ij in nodes[j].neigh:  # select my neighbours
            #     msg = nodes[j].transmit_data(ij, 'P')  # Transmit primal variable
            #     nodes[ij].receive_data(msg)  # neighbour receives the message
            # for j in range(st.n_nodes):
            nodes[j].dual_update()  # linear update of dual problem
        # for j in range(st.n_nodes):
        #     for ij in nodes[j].neigh:  # select my neighbours
        #         msg = nodes[j].transmit_data(ij, 'D')  # Transmit Dual variable
        #         nodes[ij].receive_data(msg)  # neighbour receives the message
        # for j in range(st.n_nodes):
        #     nodes[j].reorder_s_init(state)
        #     nodes[j].update('2')  # Update primal solution and state evolution
        # if i%100 == 0 and i > 0:
        #     s_hist_merged = [
        #         sum(([node.s_history[i][0][0]] for node in nodes), [])
        #         for ll in range(len(nodes[0].s_history))
        #     ]
        #     s_hist_merged = [[[], s_k] for s_k in s_hist_merged]
        #     save_snapshots(
        #         s_hist_merged,
        #         goals,
        #         None,
        #         st.dt,
        #         [(i - 1) * st.dt],
        #         f'{out_dir}/snapshot_{i}',
        #         x_lim=[-10, 10],
        #         y_lim=[-10, 10],
        #         flags=flags,
        #     )
        if np.all(np.abs(np.array(state)[:, :2] - gg) < 1e-2) and step_goal == 0:
            last_step = i + 1
            for j in range(n_robot):
                nodes[j].s_history = nodes[j].s_history[:last_step]
            time_goal = time.time() - time_start
            step_goal = i
            break
        b.update(i)

    time_elapsed = time.time() - time_start
    time_coop = time.time() - start_time_coop
    print(f'The time elapsed is {time_elapsed} seconds')
    print(f'Time used to coordinate the network is {time_coop}')
    print('The time was used in the following phases:')
    tot_creation = 0
    tot_solve = 0
    max_value = []
    creation = []
    for n, agent in enumerate(nodes):
        max_key_len = max(map(len, agent.hompc.solve_times.keys()))
        for key, value in agent.hompc.solve_times.items():
            key_len = len(key)
            if key == 'Create Problem':
                tot_creation += value
                creation.append(value)
            if key == 'Solve Problem':
                tot_solve += value
                max_value.append(value)
            # print(f"agent{n} {key}: {' '*(max_key_len-key_len)}{value}")
    print(f'Total creation time is {tot_creation}s')
    print(f'Total solving time is {tot_solve}s')
    max_a = max(max_value)
    with open(f'{out_dir}time.txt', 'w') as file:
        file.write(
            f'dt: {st.dt}\n n_c: {st.n_control}\ntime elapsed: {time_elapsed}s\ntotal solving {tot_solve}\n time to goal: {time_goal} at iter {step_goal} \nmax {max_a}\nall max {max_value}\n cr {creation}'
        )

    if st.simulation:
        """robot_pairs = list(combinations(range(num_robots), 2))
        x = np.arange(1, last_step + 1) * st.dt
        plt.figure(figsize=(10, 6))
        for i, dist_list in enumerate(pairwise_distances):
            plt.plot(x, dist_list, label=f'Robots {robot_pairs[i]}')
        plt.axhline(y=2, color='green', lw=3, linestyle='--')
        plt.title('Time Evolution of Pairwise Robot Distances')
        plt.xlabel('Time Step')
        plt.ylabel('Distance')
        # plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{out_dir}/distances.pdf', bbox_inches='tight', format='pdf')
        plt.close()"""

        # ---------------------------------------------------------------------------- #
        #                          plot the states evolutions                          #
        # ---------------------------------------------------------------------------- #
        s_hist_merged = [
            sum(([node.s_history[i][0][0]] for node in nodes), [])
            for i in range(len(nodes[0].s_history))
        ]
        if st.type == 'omni':
            s_hist_merged = [[[], s_k] for s_k in s_hist_merged]
        elif st.type == 'uni':
            s_hist_merged = [[s_k, []] for s_k in s_hist_merged]

        s_history_all = []
        for r in range(len(nodes[0].state_k)):
            s = []
            for i in range(num_robots):
                s_k = []
                for k in range(1, st.n_control):
                    s_k.append(nodes[i].state_k[r][k])
                s.append(s_k)
            if st.type == 'omni':
                s_history_all.append([[], s])
            elif st.type == 'uni':
                s_history_all.append([s, []])

        """distances = [[] for n in range(st.n_nodes)]
        for iter in s_hist_merged:
            for nn, ag in enumerate(iter[0]):
                dist_opt = np.linalg.norm(ag[:2] - centr_sol[nn])
                distances[nn].append(dist_opt)

        # distances = np.array(distances)  # shape: (n_valid_iterations, 4)

        plt.figure(figsize=(8, 5))
        for nn, fig in enumerate(distances):
            plt.semilogy(fig, label=f'agent{nn}')
        plt.xlabel('Iteration')
        plt.ylabel('Distance (log scale)')
        plt.title('Distances per vector')
        # plt.legend()
        plt.grid(True, which='both', ls='--')
        plt.savefig(f'{out_dir}/dist_to_opt.pdf', bbox_inches='tight', format='pdf')
        plt.close()"""

        flags = MultiRobotArtistFlags()
        flags.future_trajectory = False
        flags.voronoi = False
        flags.legend = False
        # flags.centroid = False

        # save_snapshots(
        #     s_hist_merged,
        #     goals,
        #     None,
        #     st.dt,
        #     [(last_step/2 - 1) * st.dt],
        #     f'{out_dir}/snapshot',
        #     x_lim=[-7, 7],
        #     y_lim=[-7, 7],
        #     flags=flags,
        # )

        # save_snapshots(
        #     s_hist_merged,
        #     goals,
        #     None,
        #     st.dt,
        #     [(last_step - 1) * st.dt],
        #     f'{out_dir}/snapshot',
        #     x_lim=[-7, 7.5],
        #     y_lim=[-7, 7.5],
        #     flags=flags,
        # )

        display_animation(
            s_hist_merged,
            s_history_all,
            goals,
            None,
            st.dt,
            st.visual_method,
            video_name=f'{out_dir}/video.mp4',
            x_lim=[-10, 10],
            y_lim=[-10, 10],
            flags=flags,
        )
        # plot_distances(
        #     s_hist_merged,
        #     0.05,  # dt
        #     2,  # dmin
        #     f'{out_dir}/distances.pdf',
        #     to_obj=False,
        #     form=False,
        # )
    b.finish()


if __name__ == '__main__':
    for i in range(1):
        n_robot = 20
        for i in range(1):
            main(n_robot)
            n_robot += 10
