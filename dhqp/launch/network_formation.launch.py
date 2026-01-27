import copy
import os
import time
from datetime import datetime
from itertools import combinations

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from scipy.spatial.distance import pdist

import dhqp.settings as st

# from dhqp.disp_het_multi_rob import (
#     MultiRobotArtistFlags,
#     display_animation,
#     save_snapshots,
# )
from hierarchical_optimization_mpc.utils.robot_models import (
    get_omnidirectional_model,
    get_unicycle_model,
)

# from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    MAXITERS = st.n_steps  # Max iterations
    COMM_TIME = 1e-3  # communication time period

    model = {
        'unicycle': get_unicycle_model(st.dt),
        'omnidirectional': get_omnidirectional_model(st.dt),
    }

    # ---------------------------------------------------------------------------- #
    #               Create the network and connection between agents               #
    # ---------------------------------------------------------------------------- #

    # deterministic graphs
    if st.n_nodes == 1:
        graph_matrix = np.array([[0.0]])
        network_graph = nx.from_numpy_array(graph_matrix, nodelist=[0])
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
                [0.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 0.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 0.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 0.0],
            ]
        )
        network_graph = nx.from_numpy_array(graph_matrix, nodelist=[0, 1, 2, 3, 4])

    if st.n_nodes == 6:
        graph_matrix = np.array(
            [
                [0.0, 1.0, 0.0, 0.0, 1.0, 1.0],
                [1.0, 0.0, 1.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0, 1.0, 1.0],
                [1.0, 0.0, 0.0, 1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
            ]
        )
        network_graph = nx.from_numpy_array(graph_matrix, nodelist=[0, 1, 2, 3, 4, 5])
    # graph_matrix = np.zeros((st.n_nodes, st.n_nodes))

    package_name = 'dhqp'
    workspace_dir = f'{get_package_share_directory(package_name)}/../../../..'
    out_dir = f'{workspace_dir}/out/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}-dhqp/'
    os.makedirs(out_dir, exist_ok=True)

    launch_description = []  # append here your nodes

    launch_description.append(
        Node(
            package='dhqp',
            namespace='watcher',
            executable='node_graph',
            parameters=[
                {
                    'max_iters': MAXITERS,
                    'communication_time': COMM_TIME,
                    'dt': st.dt,
                    'out_dir': out_dir,
                    'N_AGENTS': st.n_nodes,
                }
            ],
            output='screen',
            prefix='xterm -title "PLOTTING AGENT" -hold -e',
        )
    )
    # Create an agents of the same type for each node of the system
    for i in range(st.n_nodes):
        nn = graph_matrix[i].flatten().tolist()
        launch_description.append(
            Node(
                package='dhqp',
                namespace=f'node_{i}',
                executable='node_i',
                parameters=[
                    {
                        'agent_id': i,
                        'max_iters': MAXITERS,
                        'communication_time': COMM_TIME,
                        'neigh': nn,
                        'dt': st.dt,
                        #'system_tasks' : s_i, #system_tasks[f'agent_{i}'],
                        #'neigh_tasks' : n_i, #neigh_tasks[f'agent_{i}'],
                        #'goals' : goals,
                        'out_dir': out_dir,
                        'N_AGENTS': st.n_nodes,
                    }
                ],
                output='screen',
                prefix=f'xterm -title "agent_{i}" -hold -e',
            )
        )

    return LaunchDescription(launch_description)
