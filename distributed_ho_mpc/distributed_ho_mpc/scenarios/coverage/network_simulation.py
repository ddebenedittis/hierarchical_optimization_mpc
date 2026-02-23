import copy
import os
import time
from datetime import datetime

import networkx as nx
import numpy as np
import progressbar
from ament_index_python.packages import get_package_share_directory
from scipy.spatial.distance import pdist

import distributed_ho_mpc.scenarios.coverage.settings as st
from distributed_ho_mpc.scenarios.coverage.node import Node
from hierarchical_optimization_mpc.utils.disp_het_multi_rob import (
    MultiRobotArtistFlags,
    display_animation,
    save_snapshots,
)
from hierarchical_optimization_mpc.utils.robot_models import (
    get_omnidirectional_model,
    get_unicycle_model,
)


def main(model_name):
    model = None
    print(f'model name: {model_name}')
    if model_name == 'omni':
        model = get_omnidirectional_model(dt=st.dt)
    elif model_name == 'unicycle':
        model = get_unicycle_model(dt=st.dt)
    elif model_name == 'heterogeneous':
        model = [get_unicycle_model(dt=st.dt), get_omnidirectional_model(dt=st.dt)]
    np.random.seed(1)
    b = progressbar.ProgressBar(maxval=st.n_steps)
    b.start()

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
            closest_neighbors = set(idx for idx, _ in distances[:5])

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
    fleet = ['o', 'u', 'o', 'u', 'o', 'u', 'o', 'u', 'o', 'u']
    goals = [np.array([0, 0]), np.array([0, 0]), np.array([0, 0])]

    time_start = time.time()

    goals = [
        np.array([5, -6]),
        np.array([-5, -6]),
        np.array([-5, 6]),
        np.array([5, 6]),
        np.array([8, 3]),
        np.array([-8, 3]),
        np.array([8, -3]),
        np.array([-8, -3]),
    ]

    system_tasks = {
        'agent_0': [
            {'prio': 1, 'name': 'input_limits'},
            # {'prio': 2, 'name': 'input_smooth'},
            {'prio': 2, 'name': 'collision_avoidance'},
            {'prio': 3, 'name': 'coverage'},
        ],
        'agent_1': [
            {'prio': 1, 'name': 'input_limits'},
            # {'prio': 2, 'name': 'input_smooth'},
            {'prio': 2, 'name': 'collision_avoidance'},
            {'prio': 3, 'name': 'coverage'},
        ],
        'agent_2': [
            {'prio': 1, 'name': 'input_limits'},
            # {'prio': 2, 'name': 'input_smooth'},
            {'prio': 2, 'name': 'collision_avoidance'},
            {'prio': 3, 'name': 'coverage'},
        ],
        'agent_3': [
            {'prio': 1, 'name': 'input_limits'},
            # {'prio': 2, 'name': 'input_smooth'},
            {'prio': 2, 'name': 'collision_avoidance'},
            {'prio': 3, 'name': 'coverage'},
        ],
        'agent_7': [
            {'prio': 1, 'name': 'input_limits'},
            # {'prio': 2, 'name': 'input_smooth'},
            {'prio': 2, 'name': 'collision_avoidance'},
            {'prio': 3, 'name': 'coverage'},
        ],
        'agent_4': [
            {'prio': 1, 'name': 'input_limits'},
            # {'prio': 2, 'name': 'input_smooth'},
            {'prio': 2, 'name': 'collision_avoidance'},
            {'prio': 3, 'name': 'coverage'},
        ],
        'agent_5': [
            {'prio': 1, 'name': 'input_limits'},
            # {'prio': 2, 'name': 'input_smooth'},
            {'prio': 2, 'name': 'collision_avoidance'},
            {'prio': 3, 'name': 'coverage'},
        ],
        'agent_6': [
            {'prio': 1, 'name': 'input_limits'},
            # {'prio': 2, 'name': 'input_smooth'},
            {'prio': 2, 'name': 'collision_avoidance'},
            {'prio': 3, 'name': 'coverage'},
        ],
        'agent_8': [
            {'prio': 1, 'name': 'input_limits'},
            # {'prio': 2, 'name': 'input_smooth'},
            {'prio': 2, 'name': 'collision_avoidance'},
            {'prio': 3, 'name': 'coverage'},
        ],
        'agent_9': [
            {'prio': 1, 'name': 'input_limits'},
            # {'prio': 2, 'name': 'input_smooth'},
            {'prio': 2, 'name': 'collision_avoidance'},
            {'prio': 3, 'name': 'coverage'},
        ],
    }

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
    if st.n_nodes == 10:
        graph_matrix = np.array(
            [
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
                [1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
            ]
        )

        graph_matrix = np.ones((10, 10)) - np.eye(10)
        # network_graph = nx.from_numpy_array(graph_matrix, nodelist=[0, 1, 2, 3, 4])
    graph_matrix = np.zeros((st.n_nodes, st.n_nodes))

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
    for i in range(st.n_nodes):
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
    out_dir = (
        f'{workspace_dir}/out/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}-coverage_{model_name}/'
    )
    os.makedirs(out_dir, exist_ok=True)

    # Create an agents of the same type for each node of the system
    for i in range(st.n_nodes):
        node = Node(
            i,  # ID
            graph_matrix[i],  # Neighbours
            fleet,  # robot model
            st.dt,  # time step
            system_tasks[f'agent_{i}'],  # agent's tasks
            neigh_tasks[f'agent_{i}'],  # neighbours tasks
            goals,  # goals to be reached
            st.n_steps,  # max simulation steps
            out_dir=out_dir,
        )
        nodes.append(node)

        # create frameworks for the agents
        nodes[i].Tasks()
        nodes[i].MPC()

    # ---------------------------------------------------------------------------- #
    #     iterate through the nodes, transmitting datas and the receiving them     #
    # ---------------------------------------------------------------------------- #

    # DISTANCES BETWEEN AGENTS
    state = [None] * st.n_nodes  # list of x for inizialization of optimization

    num_robots = st.n_nodes
    num_pairs = int(num_robots * (num_robots - 1) / 2)

    # Initialize one list per robot pair
    pairwise_distances = [[] for _ in range(num_pairs)]
    gg = np.array(
        [
            [5, -6],
            [-5, -6],
            [-5, 6],
            [5, 6],
            [8, 3],
            [-8, 3],
            [8, -3],
            [-8, -3],
        ]
    )

    start_time_coop = time.time()

    for idx, j in enumerate(fleet):
        state[idx] = nodes[idx].s.omni[0] if j == 'o' else nodes[idx].s.uni[0]
    for i in range(st.n_steps):
        if i == st.n_steps - 1:
            last_step = i + 1
        if i == 30:
            None
        if i > 0:
            neigh_connection(state, nodes, graph_matrix, st.communication_range)
        for j in range(st.n_nodes):
            nodes[j].reorder_s_init(state)
            nodes[j].update()  # Update primal solution and state evolution
        for idx, j in enumerate(fleet):
            if j == 'o':
                state[idx] = nodes[idx].s.omni[0]
            else:
                state[idx] = nodes[idx].s.uni[0]
            # for ij in nodes[j].neigh:  # select my neighbours
            #     msg = nodes[j].transmit_data(ij, 'P')  # Transmit primal variable
            #     nodes[ij].receive_data(msg)  # neighbour receives the message
            # for j in range(st.n_nodes):
            nodes[idx].dual_update()
        b.update(i)

    time_elapsed = time.time() - time_start
    time_coop = time.time() - start_time_coop
    print(f'The time elapsed is {time_elapsed} seconds')
    print(f'Time used to coordinate the network is {time_coop}')
    print('The time was used in the following phases:')
    tot_creation = 0
    tot_solve = 0
    for n, agent in enumerate(nodes):
        max_key_len = max(map(len, agent.hompc.solve_times.keys()))
        for key, value in agent.hompc.solve_times.items():
            key_len = len(key)
            if key == 'Create Problem':
                tot_creation += value
            if key == 'Solve Problem':
                tot_solve += value
            # print(f"agent{n} {key}: {' '*(max_key_len-key_len)}{value}")
    print(f'Total creation time is {tot_creation}s')
    print(f'Total solving time is {tot_solve}s')

    if st.simulation:
        # ---------------------------------------------------------------------------- #
        #                          plot the states evolutions                          #
        # ---------------------------------------------------------------------------- #
        # s_hist_merged = [
        #     sum(([node.s_history[i][0][0]] for node in nodes), [])
        #     for i in range(len(nodes[0].s_history))
        # ]

        # if model_name == 'unicycle':
        #     s_hist_merged = [[s_k, []] for s_k in s_hist_merged]
        # elif model_name == 'omni':
        #     s_hist_merged = [[[], s_k] for s_k in s_hist_merged]

        s_hist_merged = []
        for n in range(len(nodes[0].s_history)):
            row = [[], []]
            for j, node in enumerate(nodes):
                if fleet[j] == 'o':
                    row[1].append(node.s_history[n])
                else:
                    row[0].append(node.s_history[n])
            s_hist_merged.append(row)

        flags = MultiRobotArtistFlags()
        flags.centroid = False
        flags.future_trajectory = False

        save_snapshots(
            s_hist_merged,
            None,
            None,
            st.dt,
            [(last_step - 1) * st.dt, (last_step - 1) / 2 * st.dt],
            f'{out_dir}/snapshot',
            flags=flags,
        )

        display_animation(
            s_hist_merged,
            s_hist_merged,
            None,
            None,
            st.dt,
            st.visual_method,
            video_name=f'{out_dir}/video.mp4',
            flags=flags,
        )
    b.finish()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        type=str,
        choices=['omni', 'unicycle', 'heterogeneous'],
        default='heterogeneous',
        help='Model type to use for simulation',
    )
    args = parser.parse_args()

    main(model_name=args.model)
