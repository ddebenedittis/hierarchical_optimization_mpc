import numpy as np

# ---------------------------------------------------------------------------- #
#                               Network settings                               #
# ---------------------------------------------------------------------------- #
experiment_name = 'radial_swt'  # 'obst_avoid' 'form'

if experiment_name == 'radial_swt':
    goals = [
        np.array([5, 5]),
        np.array([-5, 5]),
        np.array([-5, -5]),
        np.array([5, -5]),
    ]
    n_nodes = 4
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
elif experiment_name == 'form':
    n_nodes = 4
    goals = [
        np.array([5, 5]),
        np.array([-5, 5]),
        np.array([8, 8]),
        np.array([5, -5]),
    ]
    system_tasks = {
        'agent_0': [
            {'prio': 1, 'name': 'input_limits'},
            {'prio': 2, 'name': 'input_smooth'},
            {'prio': 4, 'name': 'formation', 'agents': [[0, 1]], 'distance': 4},
            {'prio': 4, 'name': 'formation', 'agents': [[0, 2]], 'distance': 4},
            {'prio': 4, 'name': 'formation', 'agents': [[0, 3]], 'distance': 5.65},
            # {'prio': 4, 'name': 'position', 'goal': goals[0], 'goal_index': 0},
        ],
        'agent_1': [
            {'prio': 1, 'name': 'input_limits'},
            {'prio': 2, 'name': 'input_smooth'},
            {'prio': 4, 'name': 'formation', 'agents': [[1, 0]], 'distance': 4},
            {'prio': 4, 'name': 'formation', 'agents': [[1, 2]], 'distance': 4},
            # {'prio': 4, 'name': 'position', 'goal': goals[1], 'goal_index': 1},
            # {'prio': 3, 'name': 'collision_avoidance'},
        ],
        'agent_2': [
            {'prio': 1, 'name': 'input_limits'},
            {'prio': 2, 'name': 'input_smooth'},
            {'prio': 3, 'name': 'obstacles_avoidance'},
            {'prio': 4, 'name': 'formation', 'agents': [[2, 0]], 'distance': 4},
            {'prio': 4, 'name': 'formation', 'agents': [[2, 1]], 'distance': 5.65},
            {'prio': 4, 'name': 'formation', 'agents': [[2, 3]], 'distance': 4},
            {'prio': 3, 'name': 'position', 'goal': goals[2], 'goal_index': 2},
        ],
        'agent_3': [
            {'prio': 1, 'name': 'input_limits'},
            {'prio': 2, 'name': 'input_smooth'},
            {'prio': 4, 'name': 'formation', 'agents': [[3, 2]], 'distance': 4},
            {'prio': 4, 'name': 'formation', 'agents': [[3, 1]], 'distance': 4},
            {'prio': 4, 'name': 'formation', 'agents': [[0, 3]], 'distance': 5.65},
        ],
    }
elif experiment_name == 'obst_avoid':
    n_nodes = 1
    goals = [
        np.array([5, 5]),
        np.array([-5, 5]),
        np.array([8, 8]),
        np.array([5, -5]),
    ]
    system_tasks = {
        'agent_0': [
            {'prio': 1, 'name': 'input_limits'},
            {'prio': 2, 'name': 'input_smooth'},
            {'prio': 3, 'name': 'obstacle_avoidance'},
            {'prio': 4, 'name': 'position', 'goal': goals[0], 'goal_index': 0},
        ],
        'agent_1': [
            {'prio': 1, 'name': 'input_limits'},
            {'prio': 2, 'name': 'input_smooth'},
            {'prio': 4, 'name': 'formation', 'agents': [[1, 0]], 'distance': 4},
            {'prio': 4, 'name': 'formation', 'agents': [[1, 2]], 'distance': 4},
            # {'prio': 4, 'name': 'position', 'goal': goals[1], 'goal_index': 1},
            # {'prio': 3, 'name': 'collision_avoidance'},
        ],
        'agent_2': [
            {'prio': 1, 'name': 'input_limits'},
            {'prio': 2, 'name': 'input_smooth'},
            {'prio': 3, 'name': 'obstacles_avoidance'},
            {'prio': 4, 'name': 'formation', 'agents': [[2, 0]], 'distance': 4},
            {'prio': 4, 'name': 'formation', 'agents': [[2, 1]], 'distance': 5.65},
            {'prio': 4, 'name': 'formation', 'agents': [[2, 3]], 'distance': 4},
            {'prio': 3, 'name': 'position', 'goal': goals[2], 'goal_index': 2},
        ],
        'agent_3': [
            {'prio': 1, 'name': 'input_limits'},
            {'prio': 2, 'name': 'input_smooth'},
            {'prio': 4, 'name': 'formation', 'agents': [[3, 2]], 'distance': 4},
            {'prio': 4, 'name': 'formation', 'agents': [[3, 1]], 'distance': 4},
            {'prio': 4, 'name': 'formation', 'agents': [[0, 3]], 'distance': 5.65},
        ],
    }
p = 1  # probability of arc of communication
# n_nodes = 1  # numbers of nodes MOVED INSIDE IF DEPENDING ON EXPERIMENT
random_graph = False  # create a random graph or not
I_NN = np.identity(n_nodes, dtype=int)

dt = 0.02
n_steps = 1000
inner_loop = 1  # number of inner loop of the distributed algorithm

communication_range = 8

v_max = 1
v_min = 0.0
omega_max = 2.0
omega_min = -2.0

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
