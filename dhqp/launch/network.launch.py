import os
from datetime import datetime

import networkx as nx
import numpy as np
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

import dhqp.settings as st


def generate_launch_description():
    MAXITERS = st.n_steps  # Max iterations
    COMM_TIME = 1e-3  # communication time period

    # ---------------------------------------------------------------------------- #
    #               Create the network and connection between agents               #
    # ---------------------------------------------------------------------------- #

    # deterministic graphs
    graph_matrix = np.ones((st.n_nodes, st.n_nodes)) - np.eye(st.n_nodes)

    package_name = 'dhqp'
    workspace_dir = f'{get_package_share_directory(package_name)}/../../../..'
    out_dir = (
        f'{workspace_dir}/out/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}-{st.experiment_name}/'
    )
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
            # prefix='xterm -title "PLOTTING AGENT" -hold -e',
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
                        'n_xi': st.n_xi,
                        'step_size': st.step_size,
                        'n_priority': st.n_priority,
                        'n_connection': st.n_connection,
                        'out_dir': out_dir,
                        'N_AGENTS': st.n_nodes,
                    }
                ],
                output='screen',
                # prefix=f'xterm -title "agent_{i}" -hold -e',
            )
        )

    return LaunchDescription(launch_description)
