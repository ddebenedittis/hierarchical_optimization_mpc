import copy
from itertools import combinations
import networkx as nx
import numpy as np 
from scipy.spatial.distance import pdist
import settings as st
import time


from hierarchical_optimization_mpc.utils.robot_models import get_unicycle_model, get_omnidirectional_model, RobCont
#from hierarchical_optimization_mpc.utils.disp_het_multi_rob import display_animation
from ho_mpc.disp_het_multi_rob import (
    display_animation,
    MultiRobotArtists,
    plot_distances,
    save_snapshots
)

from distributed_ho_mpc.node import Node
import matplotlib.pyplot as plt

model = {
   'unicycle': get_unicycle_model(st.dt),
   'omnidirectional': get_omnidirectional_model(st.dt)
}
'''
def neigh_connection(states, nodes, graph_matrix, communication_range):
    """
        Check if the distance between two nodes is less than the communication range, 
        and if so create a connection between the two nodes which become neighbours
        and cooperate one with the other
    """

    for i in range(len(nodes)):
        for idx ,state in enumerate(states):
            if i == idx:  
                continue
            # Calculate the Euclidean distance between two points and compare with comm range
            if np.linalg.norm(states[i] - state, ord=2) < communication_range: 
                if graph_matrix[i][idx] == 0.:
                    graph_matrix[i][idx] = 1.0  # modify the graph matrix to add a connection
                
                    # update task manifold with the neighbours tasks of the new neighbours
                    neigh_tasks = {}
                    
                    
                    neigh_tasks[f'agent_{i}'] = {} 
                    
                    if graph_matrix[i][idx] != 0:
                        neigh_tasks[f'agent_{i}'][f'agent_{idx}'] = copy.deepcopy(system_tasks[f'agent_{idx}']) 
                    

                    # modify the inner structure of the node        
                    nodes[i].create_connection(
                        graph_matrix[i],  # Update the neighbours of the node
                        neigh_tasks[f'agent_{i}'],  # Update the neighbours tasks
                        states[idx] # pass state of the new neighbour
                    )
            else:
                if graph_matrix[i][idx] == 1.0:  # If the distance is greater than the communication range, remove the connection
                    graph_matrix[i][idx] = 0.0

                    # modify the inner structure of the node        
                    nodes[i].remove_connection(
                        graph_matrix[i],  # Update the neighbours of the node
                        f'agent_{idx}',  # Update the neighbours tasks
                        idx,  # Index of the neighbour to remove
                        #states[np.nonzero(graph_matrix[i])[0][0]]
                    )
'''

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
        closest_neighbors = set(idx for idx, _ in distances[:7])

        current_connections = set(np.nonzero(graph_matrix[i])[0])

        to_connect = closest_neighbors - current_connections
        to_disconnect = current_connections - closest_neighbors

        # --- CONNECT (bidirectional)
        for idx in to_connect:
            graph_matrix[i][idx] = 1.0
            graph_matrix[idx][i] = 1.0  # mirror connection

            # i connects to idx
            tasks_i = {f'agent_{i}': {f'agent_{idx}': copy.deepcopy(system_tasks[f'agent_{idx}'])}}
            nodes[i].create_connection(graph_matrix[i], tasks_i[f'agent_{i}'], states[idx])

            # idx connects to i
            tasks_j = {f'agent_{idx}': {f'agent_{i}': copy.deepcopy(system_tasks[f'agent_{i}'])}}
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
    positions_over_time = positions_over_time[:,:2]
    distances = pdist(positions_over_time, metric='euclidean')  # shape: (num_pairs,)
    for i, d in enumerate(distances):
        pairwise_distances[i].append(d)
    return pairwise_distances        
                
# =========================================================================== #
#                                TASK SCHEDULER                               #
# =========================================================================== #

goals = [
        np.array([0,0 ]),
        np.array([0,0]),
        np.array([0,0])
    ]

time_start = time.time()

goals = [
    np.array([5, -6]),
    np.array([-5, -6]),
    np.array([-5, 6]),
    np.array([5, 6]),
    np.array([8,3]),
    np.array([-8,3]),
    np.array([8,-3]),
    np.array([-8,-3]),
]

'''system_tasks = {
    'agent_0': [{'prio':1, 'name':"input_limits"},
                {'prio':2, 'name':"input_smooth"},
                {'prio':3, 'name':"position", 'goal': goals[0],'goal_index':0,},
                #{'prio':3, 'name':"formation", 'agents': [[0,1]]},
                ],
    'agent_1': [{'prio':1, 'name':"input_limits"},
                {'prio':2, 'name':"input_smooth"},
                {'prio':3, 'name':"formation", 'agents': [[0,1],[1,2]]},
                #{'prio':4, 'name':"position", 'goal': goals[1],'goal_index':1,}
                ],
    'agent_2': [{'prio':1, 'name':"input_limits"},
                {'prio':2, 'name':"input_smooth"},
                {'prio':3, 'name':"position", 'goal': goals[1],'goal_index':1,},
                #{'prio':4, 'name':"formation", 'agents': [[2,3]]},
                ],
    'agent_3': [{'prio':1, 'name':"input_limits"},
                {'prio':2, 'name':"input_smooth"},
                {'prio':4, 'name':"position", 'goal': goals[1],'goal_index':1,},
                {'prio':3, 'name':"formation", 'agents': [[2,3]]},
                ]
}'''
'''system_tasks = {
    'agent_0': [{'prio':1, 'name':"input_limits"},
                {'prio':2, 'name':"input_smooth"},
                #{'prio':4, 'name':"obstacle_avoidance"},
                {'prio':3, 'name':"position", 'goal': goals[0],'goal_index':0,},
                {'prio':4, 'name':"position", 'goal': goals[1],'goal_index':1},
                ],
    'agent_1': [{'prio':1, 'name':"input_limits"},
                {'prio':2, 'name':"input_smooth"},
                {'prio':3, 'name':"collision_avoidance"},
                {'prio':4, 'name':"formation", 'agents': [[0,1]], 'distance': 3},
                #{'prio':4, 'name':"position", 'goal': goals[0],'goal_index':0,},
                ],
    'agent_2': [{'prio':1, 'name':"input_limits"},
                {'prio':2, 'name':"input_smooth"},
                #{'prio':3, 'name':"collision_avoidance"},
                {'prio':3, 'name':"formation", 'agents': [[2,3]], 'distance': 5},
                {'prio':4, 'name':"formation", 'agents': [[1,2]], 'distance': 2}
                ],
    'agent_3': [{'prio':1, 'name':"input_limits"},
                {'prio':2, 'name':"input_smooth"},
                #{'prio':3, 'name':"collision_avoidance"},
                {'prio':3, 'name':"position", 'goal': goals[1],'goal_index':1,},
                {'prio':4, 'name':"formation", 'agents': [[3,4]], 'distance': 3}
                ],
    'agent_4': [{'prio':1, 'name':"input_limits"},
                {'prio':2, 'name':"input_smooth"},
                #{'prio':3, 'name':"collision_avoidance"},
                {'prio':3, 'name':"formation", 'agents': [[3,4]], 'distance': 5},
                {'prio':4, 'name':"position", 'goal': goals[1],'goal_index':1,},
                ]
}'''
system_tasks = {'agent_0': [{'prio':1, 'name':"input_limits"},
                            {'prio':2, 'name':"input_smooth"},
                            {'prio':3, 'name':"coverage"},
                ],
                'agent_1': [{'prio':1, 'name':"input_limits"},
                            {'prio':2, 'name':"input_smooth"},
                            {'prio':3, 'name':"coverage"},
                ],
                'agent_2': [{'prio':1, 'name':"input_limits"},
                            {'prio':2, 'name':"input_smooth"},
                            {'prio':3, 'name':"coverage"},
                ],
                'agent_3': [{'prio':1, 'name':"input_limits"},
                            {'prio':2, 'name':"input_smooth"},
                            {'prio':3, 'name':"coverage"},
                ],
                'agent_4': [{'prio':1, 'name':"input_limits"},
                            {'prio':2, 'name':"input_smooth"},
                            {'prio':3, 'name':"collision_avoidance"},
                            #{'prio':3, 'name':"formation", 'agents': [[0,3]], 'distance': 4},
                            #{'prio':4, 'name':"position", 'goal': goals[3],'goal_index':3},
                ],
                'agent_4': [{'prio':1, 'name':"input_limits"},
                            {'prio':2, 'name':"input_smooth"},
                            {'prio':3, 'name':"collision_avoidance"},
                            #{'prio':3, 'name':"formation", 'agents': [[0,3]], 'distance': 4},
                            #{'prio':4, 'name':"position", 'goal': goals[4],'goal_index':4},
                ],
                'agent_5': [{'prio':1, 'name':"input_limits"},
                            {'prio':2, 'name':"input_smooth"},
                            {'prio':3, 'name':"collision_avoidance"},
                            #{'prio':3, 'name':"formation", 'agents': [[0,3]], 'distance': 4},
                            #{'prio':4, 'name':"position", 'goal': goals[5],'goal_index':5},
                ],
                'agent_6': [{'prio':1, 'name':"input_limits"},
                            {'prio':2, 'name':"input_smooth"},
                            {'prio':3, 'name':"collision_avoidance"},
                            #{'prio':3, 'name':"formation", 'agents': [[0,3]], 'distance': 4},
                            #{'prio':4, 'name':"position", 'goal': goals[6],'goal_index':6},
                ],
                'agent_7': [{'prio':1, 'name':"input_limits"},
                            {'prio':2, 'name':"input_smooth"},
                            {'prio':3, 'name':"collision_avoidance"},
                            #{'prio':3, 'name':"formation", 'agents': [[0,3]], 'distance': 4},
                ],
}


# ---------------------------------------------------------------------------- #
#               Create the network and connection between agents               #
# ---------------------------------------------------------------------------- #

# deterministic graphs
if st.n_nodes == 2:
    graph_matrix = np.array([[0.,1.],
                             [1.,0.]])
    network_graph = nx.from_numpy_array(graph_matrix, nodelist = [0,1])
if st.n_nodes == 3:
    graph_matrix = np.array([[0., 1., 1.],
                             [1., 0., 1.],
                             [1., 1., 0.]])
    network_graph = nx.from_numpy_array(graph_matrix, nodelist = [0,1,2])
if st.n_nodes == 4:
    graph_matrix = np.array([[0., 1., 1., 0.],
                             [1., 0., 1., 0.],
                             [1., 1., 0., 1.],
                             [0., 0., 1., 0.]])
    network_graph = nx.from_numpy_array(graph_matrix, nodelist = [0,1,2,3])
if st.n_nodes == 5:
    graph_matrix = np.array([[0., 1., 0., 0., 0.],
                             [1., 0., 1., 0., 0.],
                             [0., 1., 0., 1., 0.],
                             [0., 0., 1., 0., 1.],
                             [0., 0., 0., 1., 0.]])
    network_graph = nx.from_numpy_array(graph_matrix, nodelist = [0,1,2,3,4])
#graph_matrix = np.zeros((st.n_nodes, st.n_nodes)) 


# random graph 🎲
while st.random_graph:
    network_graph = nx.gnp_random_graph(st.n_nodes, st.p)
    graph_matrix = nx.to_numpy_array(network_graph)

    test = np.linalg.matrix_power((st.I_NN+graph_matrix),st.n_nodes)

    if np.all(test>0):
        print("the graph is connected")
        #nx.draw(network_graph)
        #plt.show()
        break 
    else:
        print("the graph is NOT connected")

# update task manifold with the neighbours tasks
neigh_tasks = {}
for i in range(st.n_nodes):
    id = 0
    neigh_tasks[f'agent_{i}'] = {} 
    for j in graph_matrix[i]:
        if int(j) != 0:
            neigh_tasks[f'agent_{i}'][f'agent_{id}'] = copy.deepcopy(system_tasks[f'agent_{id}']) 
        id += 1
                                                         
#----------------------------------------------------------------------------- #
#         Create agents and initialize them based on settings and tasks        #
# ---------------------------------------------------------------------------- #

nodes = [] # list of agents of the system

#Create an agents of the same type for each node of the system
for i in range(st.n_nodes):
    node = Node(i,                   # ID
        graph_matrix[i],             # Neighbours
        model['unicycle'],           # robot model
        st.dt,                       # time step
        system_tasks[f'agent_{i}'],  # agent's tasks
        neigh_tasks[f'agent_{i}'],   # neighbours tasks
        goals,                       # goals to be reached
        st.n_steps                   # max simulation steps
    )
    nodes.append(node)

    # create frameworks for the agents
    nodes[i].Tasks()
    nodes[i].MPC()


# ---------------------------------------------------------------------------- #
#     iterate through the nodes, transmitting datas and the receiving them     #
# ---------------------------------------------------------------------------- #

# DISTANCES BETWEEN AGENTS
state = [None] * st.n_nodes # list of x for inizialization of optimization

num_robots = st.n_nodes
num_pairs = int(num_robots * (num_robots - 1) / 2)

# Initialize one list per robot pair
pairwise_distances = [[] for _ in range(num_pairs)]
gg = np.array([
        [5, -6],
        [-5, -6],
        [-5, 6],
        [5, 6],
        [8,3],
        [-8,3],
        [8,-3],
        [-8,-3],
    ])



start_time_coop = time.time()

for j in range(st.n_nodes):
    state[j] = nodes[j].s.omni[0] # TODO manage heterogeneous robots
#neigh_connection(state, nodes, graph_matrix, st.communication_range) 
for j in range(st.n_nodes):
    nodes[j].reorder_s_init(state)
    nodes[j].update()    # Update primal solution and state evolution
for j in range(st.n_nodes):
    state[j] = nodes[j].s.omni[0] # TODO manage heterogeneous robots
    for ij in nodes[j].neigh:  # select my neighbours
        msg = nodes[j].transmit_data(ij, 'P') # Transmit primal variable
        nodes[ij].receive_data(msg) # neighbour receives the message
for j in range(st.n_nodes):
    nodes[j].dual_update()    # linear update of dual problem
    
for i in range(st.n_steps):
    # if np.all(np.abs(np.array(state)[:,:2] - gg) < 10e-3):
    #     last_step = i
    #     break
    if i == 37:
        None
    if i == st.n_steps-1:
        last_step = i+1
    neigh_connection(state, nodes, graph_matrix, st.communication_range) 
    for j in range(st.n_nodes):
        for ij in nodes[j].neigh:  # select my neighbours
            msg = nodes[j].transmit_data(ij, 'D') # Transmit Dual variable
            nodes[ij].receive_data(msg) # neighbour receives the message
    for j in range(st.n_nodes):
        nodes[j].reorder_s_init(state) 
        nodes[j].update()    # Update primal solution and state evolution
    for j in range(st.n_nodes):
        state[j] = nodes[j].s.omni[0] # TODO manage heterogeneous robots
        for ij in nodes[j].neigh:  # select my neighbours
            msg = nodes[j].transmit_data(ij, 'P') # Transmit primal variable
            nodes[ij].receive_data(msg) # neighbour receives the message
    for j in range(st.n_nodes):
        nodes[j].dual_update()    # linear update of dual problem
    pairwise_distances = agents_distance(state, pairwise_distances)


time_elapsed = time.time() - time_start
time_coop = time.time() - start_time_coop
print(f"The time elapsed is {time_elapsed} seconds")
print(f'Time used to coordinate the network is {time_coop}')
print( "The time was used in the following phases:")
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
        #print(f"agent{n} {key}: {' '*(max_key_len-key_len)}{value}")
print(f'Total creation time is {tot_creation}s')
print(f'Total solving time is {tot_solve}s')


if st.simulation:
    '''robot_pairs = list(combinations(range(num_robots), 2))
    x = np.arange(1, last_step+1) * st.dt
    plt.figure(figsize=(10, 6))
    for i, dist_list in enumerate(pairwise_distances):
        plt.plot(x, dist_list, label=f'Robots {robot_pairs[i]}')
    plt.axhline(y = 2, color='green', lw=4, linestyle='--')
    plt.title("Time Evolution of Pairwise Robot Distances")
    plt.xlabel("Time Step")
    plt.ylabel("Distance")
    #plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()'''

    # ---------------------------------------------------------------------------- #
    #                          plot the states evolutions                          #
    # ---------------------------------------------------------------------------- #
    s_hist_merged = [
        sum((node.s_history[i][:1] for node in nodes), []) for i in range(len(nodes[0].s_history))
    ]

    # handle different lenght of the states due to add/remove of nodes
    for i in nodes:
        for n in range(len(i.s_history)):
            max_len = max(len(inner_list) for outer in i.s_history for inner_list in outer)
            if len(i.s_history[n][0]) < max_len:
                dd = max_len - len(i.s_history[n][0])
                for d in range(dd): 
                    if n == 0:
                        if len(i.s_history[n+1][0]) <= dd :
                            i.s_history[n][0].append(np.array([30, 30, 0])) # value out of the limits of the simulation
                        else:
                            i.s_history[n][0].append(i.s_history[1][0][d]) # take next value
                    else:
                        i.s_history[n][0].append(i.s_history[n-1][0][d+1]) # take previous value


    #s_hist_merged = [ [[s_hist_merged[0][0],s_hist_merged[0][1]], np.array([0,0,0])] for i in s_hist_merged]
    if st.n_nodes == 2:
        s_hist_merged = [nodes[0].s_history[i] + nodes[1].s_history[i] for i in range(len(nodes[0].s_history))]
    if st.n_nodes == 3:
        s_hist_merged = [nodes[0].s_history[i] + nodes[1].s_history[i] + nodes[2].s_history[i] for i in range(len(nodes[0].s_history))]
    if st.n_nodes == 4: 
        s_hist_merged = [nodes[0].s_history[i] + nodes[1].s_history[i] + nodes[2].s_history[i] + nodes[3].s_history[i] for i in range(len(nodes[0].s_history))]
    if st.n_nodes == 5:
        s_hist_merged = [nodes[0].s_history[i] + nodes[1].s_history[i] + nodes[2].s_history[i] + nodes[3].s_history[i] + nodes[4].s_history[i] for i in range(len(nodes[0].s_history))]
        
    
    artist_flags = MultiRobotArtists(
        centroid=True, goals=True, obstacles=False,
        past_trajectory=True,
        omnidir=True,
        unicycles=False,
        #robots=RobCont(omni=True),
        #robot_names=True,
        voronoi=False,
    )


    display_animation(s_hist_merged, None, None, st.dt, st.visual_method, show_voronoi=True, show_trajectory=True, estim=st.estimation_plotting)

