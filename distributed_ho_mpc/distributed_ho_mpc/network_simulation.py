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

# =========================================================================== #
#                                TASK SCHEDULER                               #
# =========================================================================== #

goals = [
        np.array([6, -5]),
        np.array([5, 3]),
    ]

system_tasks = {
    'agent_0': [{'prio':1,             # priority
                 'name':"input_limits",   # task type
                    },
                {'prio':2,             # priority
                 'name':"input_smooth",   # task type
                      },
                {'prio':3,             # priority
                 'name':"position",   # task type
                 'goal': goals[0],         # [x,y] 
                 'goal_index':0,          # index of the corrisponding list goal's element 
                    },
                ],
    'agent_1': [{'prio':1,             # priority
                 'name':"input_limits",   # task type
                    },
                {'prio':2,             # priority
                 'name':"input_smooth",   # task type
                },
                {'prio':3,             # priority
                 'name':"formation",   # task type
                 'agents': [0,1]
                },
                ],
    'agent_2': [{'prio':1,             # priority
                 'name':"input_limits",   # task type
                },
                {'prio':2,             # priority
                 'name':"input_smooth",   # task type
                },
                {'prio':3,             # priority
                 'name':"position",   # task type
                 'goal': goals[1],         # [x,y] 
                 'goal_index':1,          # index of the corrisponding list goal's element 
                    },
                ],
    'agent_3': [{'prio':1,             # priority
                 'name':"input_limits",   # task type
                      },
                {'prio':2,             # priority
                 'name':"input_smooth",   # task type
                      },
                {'prio':3,             # priority
                 'name':"position",   # task type
                'goal': goals[0],         # [x,y]
                'goal_index':0, 
                      },
                {'prio':4,             # priority
                 'name':"position",   # task type
                'goal': goals[1],         # [x,y]
                'goal_index':0, 
                      }
                ],
}

# ---------------------------------------------------------------------------- #
#               Create the network and connection between agents               #
# ---------------------------------------------------------------------------- #


# deterministic graphs = evolve(s, u_star, dt)
'''if not st.random_graph:
    graph_matrix = np.array([[0.,1.],
                             [1.,0.]])
    network_graph = nx.from_numpy_array(graph_matrix, nodelist = [0,1])'''
if not st.random_graph:
    graph_matrix = np.array([[0.,1., 0.],
                             [1.,0., 1.],
                             [0.,1., 0.]])
    network_graph = nx.from_numpy_array(graph_matrix, nodelist = [0,1,2])


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
            #neigh_tasks[f'agent_{i}'].append({'neigh_ID': id})  
        id += 1
                                                         
#----------------------------------------------------------------------------- #
#         Create agents and initialize them based on settings and tasks        #
# ---------------------------------------------------------------------------- #

nodes = [] # list of agents of the system

#Create an agents of the same type for each node of the system
for i in range(st.n_nodes):
    node = Node(i,                          # ID
                graph_matrix[i],            # Neighbours
                model['unicycle'],          # robot model
                st.dt,                      # time step
                system_tasks[f'agent_{i}'],  # agent's tasks
                neigh_tasks[f'agent_{i}'],   # neighbours tasks
                goals,                      # goals to be reached
                st.n_steps                  # max simulation steps
                    )
    nodes.append(node)

    # create frameworks for the agents
    nodes[i].Tasks()
    nodes[i].MPC()


# ---------------------------------------------------------------------------- #
#     iterate through the nodes, transmitting datas and the receiving them     #
# ---------------------------------------------------------------------------- #

for j in range(st.n_nodes):
    nodes[j].update()    # Update primal solution and state evolution
for j in range(st.n_nodes):
    for ij in nodes[j].neigh:  # select my neighbours
        msg = nodes[j].transmit_data(ij, 'P') # Transmit primal variable
        nodes[ij].receive_data(msg) # neighbour receives the message
for j in range(st.n_nodes):
    nodes[j].dual_update()    # linear update of dual problem

for i in range(st.n_steps):
    for j in range(st.n_nodes):
        for ij in nodes[j].neigh:  # select my neighbours
            msg = nodes[j].transmit_data(ij, 'D') # Transmit Dual variable
            nodes[ij].receive_data(msg) # neighbour receives the message
    for j in range(st.n_nodes):
        nodes[j].update()    # Update primal solution and state evolution
    for j in range(st.n_nodes):
        for ij in nodes[j].neigh:  # select my neighbours
            msg = nodes[j].transmit_data(ij, 'P') # Transmit primal variable
            nodes[ij].receive_data(msg) # neighbour receives the message
    for j in range(st.n_nodes):
        nodes[j].dual_update()    # linear update of dual problem

# ---------------------------------------------------------------------------- #
#                          plot the states evolutions                          #
# ---------------------------------------------------------------------------- #
# s_hist_merged = [
#     [
#         sum((node.s_history[i][j][:1] for node in nodes), []) 
#         for j in range(len(nodes[0].s_history[i]))
#     ]
#     for i in range(len(nodes[0].s_history))
# ]
s_hist_merged = [
            sum((node.s_history[i][:1] for node in nodes), []) 
    for i in range(len(nodes[0].s_history))
]
#s_hist_merged = [ [[s_hist_merged[0][0],s_hist_merged[0][1]], np.array([0,0,0])] for i in s_hist_merged]
s_hist_merged = [nodes[0].s_history[i] + nodes[1].s_history[i] + nodes[2].s_history[i] for i in range(len(nodes[0].s_history))]


'''display_animation(nodes[0].s_history, goals, None, st.dt, st.visual_method)
display_animation(nodes[1].s_history, goals, None, st.dt, st.visual_method)'''
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


# ---------------------------------------------------------------------------- #
#            plot the comparison of dual variables of the agents               #
# ---------------------------------------------------------------------------- #
rho = [None] * st.n_nodes
idx = [None] * st.n_nodes
for i in range(st.n_nodes):
    rho[i], idx[i] =nodes[i].plot_dual()
for i in range(st.n_nodes):
    for j in idx[i]:
        for p in range(st.n_priority):
            plt.plot(rho[i][0, p, (j * st.n_xi): (j + 1) * st.n_xi], label=f'agent {i}-{j} priority {p}')
            plt.show()