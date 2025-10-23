import numpy as np

# ---------------------------------------------------------------------------- #
#                               Network settings                               #
# ---------------------------------------------------------------------------- #
p = 1  # probability of arc of communication
n_nodes = 2  # numbers of nodes
random_graph = False  # create a random graph or not
I_NN = np.identity(n_nodes, dtype=int)

dt = 0.01
n_steps = 4000
inner_loop = 1  # number of inner loop of the distributed algorithm

communication_range = 8

v_max = 1.5
v_min = -1.5
omega_max = 2
omega_min = -2

# ---------------------------------------------------------------------------- #
#                              Flags for simulation                            #
# ---------------------------------------------------------------------------- #
output = {'display': 'plot', 'save': 'save', 'nothing': 'none'}
visual_method = output['nothing']  # change the key to decide the output visualization
save_data = True
simulation = True
inner_plot = False  # plot the inner state of the robots
estimation_plotting = False
# ---------------------------------------------------------------------------- #
#                                 MPC settings                                 #
# ---------------------------------------------------------------------------- #
n_control = 1  # mpc control step
n_pred = 0  # mpc prediction step

n_xi = n_control * 2

# ---------------------------------------------------------------------------- #
#                                 PDD settings                               #
# ---------------------------------------------------------------------------- #


n_priority = 4
step_size = 1e-5

goals = [
    np.array([5, 5]),
    np.array([-5, -5]),
    np.array([-5, 5]),
    np.array([5, -5]),
]
system_tasks = {
    'agent_0': [
        {'prio': 1, 'name': 'input_limits'},
        {'prio': 2, 'name': 'input_smooth'},
        {'prio': 3, 'name': 'collision_avoidance'},
        {'prio': 4, 'name': 'position', 'goal': goals[0], 'goal_index': 0},
    ],
    'agent_1': [
        {'prio': 1, 'name': 'input_limits'},
        {'prio': 2, 'name': 'input_smooth'},
        {'prio': 4, 'name': 'position', 'goal': goals[1], 'goal_index': 1},
        {'prio': 3, 'name': 'collision_avoidance'},
    ],
    'agent_2': [
        {'prio': 1, 'name': 'input_limits'},
        {'prio': 2, 'name': 'input_smooth'},
        {'prio': 3, 'name': 'collision_avoidance'},
        # {'prio':4, 'name':"formation", 'agents': [[2,3]], 'distance': 4},
        {'prio': 4, 'name': 'position', 'goal': goals[2], 'goal_index': 2},
    ],
    'agent_3': [
        {'prio': 1, 'name': 'input_limits'},
        {'prio': 2, 'name': 'input_smooth'},
        {'prio': 3, 'name': 'collision_avoidance'},
        # {'prio':3, 'name':"formation", 'agents': [[0,3]], 'distance': 4},
        {'prio': 4, 'name': 'position', 'goal': goals[3], 'goal_index': 3},
    ],
}
