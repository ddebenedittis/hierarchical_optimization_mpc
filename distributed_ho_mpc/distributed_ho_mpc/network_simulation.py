import numpy as np 
import networkx as nx
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from Node import Node
from robot_models import get_unicycle_model, get_omnidirectional_model
import settings as st
from auxiliary.disp_het_multi_rob import display_animation

model = {
   'unicycle': get_unicycle_model(st.dt),
   'omnidirectional': get_omnidirectional_model(st.dt)
}

# ---------------------------------------------------------------------------- #
#                                 Task handler                                 #
# ---------------------------------------------------------------------------- #

goals = [
        np.array([13, -16]),
        np.array([11, 6]),
        np.array([-5, 4]),
    ]

system_tasks = {
    'agent_0': [{'prio':1,             # priority
                 'name':"input_limits",   # task type
                 'Xsym' : None
                      },
                {'prio':2,             # priority
                 'name':"input_smooth",   # task type
                 'Xsym' : None
                      },
                {'prio':3,             # priority
                 'name':"position",   # task type
                 'goal': goals[0],         # [x,y] 
                 'goal_index':0,          # index of the corrisponding list goal's element 
                 'Xsym': [[('u1',2),('s1',3)],[('u2',2),('s2',3)]]
                        },
                {'prio':4,             # priority
                 'name':"position",   # task type
                'goal': goals[1],         # [x,y]
                'goal_index':1,           # index of the corrisponding list goal's element 
                'Xsym' : [[('u2',2),('s2',3)],[('u1',2),('s1',3)]]
                      }
                ],
    'agent_1': [{'prio':1,             # priority
                 'name':"input_limits",   # task type
                 'Xsym' : None
                      },
                {'prio':2,             # priority
                 'name':"input_smooth",   # task type
                 'Xsym' : None
                      },
                {'prio':3,             # priority
                 'name':"position",   # task type
                 'goal': goals[1],         # [x,y] 
                 'goal_index':1,          # index of the corrisponding list goal's element 
                 'Xsym': [[('u1',2),('s1',3)],[('u2',2),('s2',3)]]
                        },
                {'prio':4,             # priority
                 'name':"position",   # task type
                'goal': goals[0],         # [x,y]
                'goal_index':0,           # index of the corrisponding list goal's element 
                'Xsym' : [[('u2',2),('s2',3)],[('u1',2),('s1',3)]]
                      }     
                ],
    'agent_2': [{'prio':1,             # priority
                 'name':"input_limits",   # task type
                 'Xsym' : None
                      },
                {'prio':2,             # priority
                 'name':"input_smooth",   # task type
                 'Xsym' : None
                      },
                {'prio':3,             # priority
                 'name':"position",   # task type
                 'goal': goals[2],         # [x,y] 
                 'goal_index':2,          # index of the corrisponding list goal's element 
                 'Xsym': [[('u1',2),('s1',3)],[('u2',2),('s2',3)]]
                        }     
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
if not st.random_graph:
    graph_matrix = np.array([[0.,1.,1.,0.],
                            [1.,0.,1.,0.],
                            [1.,1.,0.,1.],    
                            [0.,0.,1.,0.]])
    network_graph = nx.from_numpy_array(graph_matrix, nodelist = [0,1,2,3])


# random graph
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

# ---------------------------------------------------------------------------- #
#         Create agents and initialize them based on settings and tasks        #
# ---------------------------------------------------------------------------- #


# shared list to communicate state with the neighbours (not used)
buffer_states = [0, 0, 0, 0]

nodes = [] # list of agents of the system

#Create an agents of the same type for each node of the system
for i in range(st.n_nodes):
    node = Node(i,                          # ID
                graph_matrix[i],            # Neighbours
                model['unicycle'],          # robot model
                st.dt,                         # time step
                system_tasks[f'agent_{i}'],  # agent's tasks
                goals,                      #
                st.n_steps                  #
                    )
    nodes.append(node)

    # create frameworks for the agents
    nodes[i].Tasks()
    nodes[i].MPC()


# ---------------------------------------------------------------------------- #
#     iterate through the nodes, transmitting datas and the receiving them     #
# ---------------------------------------------------------------------------- #

for i in range(st.n_steps):
    for j in range(st.n_nodes):
        msg, nn = nodes[j].transmit_data()      # Transmit state to shared memory
        for jj in nn:
            nodes[jj].receive_data(msg)         # Neighbours receive data 
    for j in range(st.n_nodes):
        nodes[j].update()    # Update state evolution


'''# plot the states evolutions
for j in range(st.n_nodes):
    plt.plot(nodes[j].x_past, label=f'Agent_{j}')
plt.legend()
plt.show()

fig, ax = plt.subplots()
scat = ax.scatter(nodes[0].s[0], nodes[0].s[1])
ax.set(xlim=[-5, 5], ylim=[-5, 5], xlabel='x [m]', ylabel='y [m]')
ax.legend()

anim = Animation(scat, nodes[0].s_history)
temp = FuncAnimation(fig=fig, func=anim.update, frames=range(1000), interval=30)'''

s_hist_merged = [
    [
        sum((node.s_history[i][j][:1] for node in nodes), [])
        for j in range(len(nodes[0].s_history[i]))
    ]
    for i in range(len(nodes[0].s_history))
]


# display_animation(nodes[0].s_history, goals, None, st.dt, st.visual_method)
# display_animation(nodes[1].s_history, goals, None, st.dt, st.visual_method)
display_animation(s_hist_merged, goals, None, st.dt, st.visual_method, show_voronoi=False)

