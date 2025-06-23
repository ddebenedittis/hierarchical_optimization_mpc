import copy
import networkx as nx
import numpy as np 
import settings as st

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

def neigh_connection(states, nodes, graph_matrix, communication_range):
    """
        Check if the distance between two nodes is less than the communication range, 
        and if so create a connection between the two nodes which become neighbours
        and cooperate one with the other
    """

    for i, node in enumerate(nodes):
        for idx ,state in enumerate(states):
            if i == idx:  
                continue
            # Calculate the Euclidean distance between two points and compare with comm range
            if np.linalg.norm(states[i] - state) < communication_range: 
                if graph_matrix[i][idx] == 0.:
                    graph_matrix[i][idx] = 1.0  # modify the graph matrix to add a connection
                
                    # update task manifold with the neighbours tasks of the new neighbours
                    neigh_tasks = {}
                    
                    id = 0
                    neigh_tasks[f'agent_{i}'] = {} 
                    for j in graph_matrix[i]:
                        if int(j) != 0:
                            neigh_tasks[f'agent_{i}'][f'agent_{id}'] = copy.deepcopy(system_tasks[f'agent_{id}']) 
                        id += 1

                    # modify the inner structure of the node        
                    nodes[i].create_connection(
                        graph_matrix[i],  # Update the neighbours of the node
                        neigh_tasks[f'agent_{i}'],  # Update the neighbours tasks
                        states[np.nonzero(graph_matrix[i])[0][0]] # pass state of the new neighbour
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




            
                
# =========================================================================== #
#                                TASK SCHEDULER                               #
# =========================================================================== #

goals = [
        np.array([4, 5]),
        np.array([-4, -5]),
    ]

'''system_tasks = {
    'agent_0': [{'prio':1, 'name':"input_limits"},
                {'prio':2, 'name':"input_smooth"},                
                {'prio':3, 'name':"formation", 'agents': [[0,1]]},
                {'prio':4, 'name':"position", 'goal': goals[1],'goal_index':1,},
                ],
    'agent_1': [{'prio':1, 'name':"input_limits"},
                {'prio':2, 'name':"input_smooth"},                
                {'prio':4, 'name':"formation", 'agents': [[0,1]]},
                {'prio':3, 'name':"position", 'goal': goals[0],'goal_index':0,},
                ],
    'agent_2': [{'prio':1, 'name':"input_limits"},
                {'prio':2, 'name':"input_smooth"},                
                {'prio':3, 'name':"formation", 'agents': [[1,2]]},
                {'prio':4, 'name':"position", 'goal': goals[0],'goal_index':0,},
                ],
}'''
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
                            #{'prio':3, 'name':"formation", 'agents': [[0,1]], 'distance': 3},
                            {'prio':4, 'name':"position", 'goal': goals[0],'goal_index':0},
                ],
                'agent_1': [{'prio':1, 'name':"input_limits"},
                            {'prio':2, 'name':"input_smooth"},
                            #{'prio':3, 'name':"collision_avoidance"},
                            {'prio':4, 'name':"position", 'goal': goals[1],'goal_index':1},
                ],
}

# ---------------------------------------------------------------------------- #
#               Create the network and connection between agents               #
# ---------------------------------------------------------------------------- #

graph_matrix = np.zeros((st.n_nodes, st.n_nodes)) 
#deterministic graphs
# if st.n_nodes == 2:
#     graph_matrix = np.array([[0.,1.],
#                              [1.,0.]])
#     network_graph = nx.from_numpy_array(graph_matrix, nodelist = [0,1])
if st.n_nodes == 3:
    graph_matrix = np.array([[0.,1., 0.],
                             [1.,0., 1.],
                             [0.,1., 0.]])
    network_graph = nx.from_numpy_array(graph_matrix, nodelist = [0,1,2])
if st.n_nodes == 4:
    graph_matrix = np.array([[0.,1., 0., 0.],
                             [1.,0., 1., 0.],
                             [0.,1., 0., 1.],
                             [0.,0., 1., 0.]])
    network_graph = nx.from_numpy_array(graph_matrix, nodelist = [0,1,2,3])
if st.n_nodes == 5:
    graph_matrix = np.array([[0.,1., 0., 0., 0.],
                             [1.,0., 1., 0., 0.],
                             [0.,1., 0., 1., 0.],
                             [0.,0., 1., 0., 1.],
                             [0.,0., 0., 1., 0.]])
    '''graph_matrix = np.array([[0., 1., 1., 1., 1.],
                            [1., 0., 1., 1., 1.],
                            [1., 1., 0., 1., 1.],
                            [1., 1., 1., 0., 1.],
                            [1., 1., 1., 1., 0.]])'''
    network_graph = nx.from_numpy_array(graph_matrix, nodelist = [0,1,2,3,4])



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

state = [None] * st.n_nodes # list of x for inizialization of optimization

for j in range(st.n_nodes):
        state[j] = nodes[j].s.omni[0] # TODO manage heterogeneous robots
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


if st.simulation:
    # ---------------------------------------------------------------------------- #
    #                          plot the states evolutions                          #
    # ---------------------------------------------------------------------------- #
    s_hist_merged = [
        sum((node.s_history[i][:1] for node in nodes), []) for i in range(len(nodes[0].s_history))
    ]

    # handle different lenght of the states due to add/remove of nodes
    for i in nodes:
        for n in range(len(i.s_history)):
            if len(i.s_history[n][0]) < len(i.s_history[-1][0]):
                i.s_history[n][0].append(np.array([0,0]))
            elif len(i.s_history[n][0]) < len(i.s_history[2][0]):
                i.s_history[n][0].append(np.array([0,0]))

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
        past_trajectory=False,
        omnidir=True,
        unicycles=False,
        #robots=RobCont(omni=True),
        #robot_names=True,
        voronoi=False,
    )


    display_animation(s_hist_merged, goals, None, st.dt, st.visual_method, show_voronoi=False, show_trajectory=False)

