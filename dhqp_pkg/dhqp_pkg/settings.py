import numpy as np

# ---------------------------------------------------------------------------- #
#                               Network settings                               #
# ---------------------------------------------------------------------------- #
experiment_name = 'radial_swt'  # 'form' 'obst_avoid'

if experiment_name == 'radial_swt':
    goals = [
        np.array([2.1, 1.4]),
        np.array([2.1, -0.64]),
        np.array([-1.29, -0.34]),
        np.array([-1.29, 1.16]),
    ]
    n_nodes = 4
    system_tasks = {
        'agent_0': [
            {'prio': 1, 'name': 'input_limits'},
            {'prio': 2, 'name': 'input_smooth'},
            {'prio': 3, 'name': 'collision_avoidance'},
            {'prio': 3, 'name': 'obstacle_avoidance'},
            {'prio': 4, 'name': 'position', 'goal': goals[0], 'goal_index': 0},
        ],
        'agent_1': [
            {'prio': 1, 'name': 'input_limits'},
            {'prio': 2, 'name': 'input_smooth'},
            {'prio': 4, 'name': 'position', 'goal': goals[1], 'goal_index': 1},
            {'prio': 3, 'name': 'collision_avoidance'},
            {'prio': 3, 'name': 'obstacle_avoidance'},
        ],
        'agent_2': [
            {'prio': 1, 'name': 'input_limits'},
            {'prio': 2, 'name': 'input_smooth'},
            {'prio': 3, 'name': 'collision_avoidance'},
            {'prio': 3, 'name': 'obstacle_avoidance'},
            {'prio': 4, 'name': 'position', 'goal': goals[2], 'goal_index': 2},
        ],
        'agent_3': [
            {'prio': 1, 'name': 'input_limits'},
            {'prio': 2, 'name': 'input_smooth'},
            {'prio': 3, 'name': 'obstacle_avoidance'},
            {'prio': 3, 'name': 'collision_avoidance'},
            {'prio': 4, 'name': 'position', 'goal': goals[3], 'goal_index': 3},
        ],
    }
elif experiment_name == 'form':
    n_nodes = 5
    goals = [
        np.array([0, 0]),
    ]
    system_tasks = {
        'agent_0': [
            {'prio': 1, 'name': 'input_limits'},
            {'prio': 2, 'name': 'input_smooth'},
            # {'prio': 3, 'name': 'obstacle_avoidance'},
            {'prio': 4, 'name': 'formation', 'agents': [[0, 1]], 'distance': 1.5},
            {'prio': 4, 'name': 'formation', 'agents': [[0, 3]], 'distance': 1.5},
            {'prio': 4, 'name': 'formation', 'agents': [[0, 4]], 'distance': 2.12},
            # {'prio': 3, 'name': 'formation', 'agents': [[0, 2]], 'distance': 4},
            # {'prio': 4, 'name': 'position', 'goal': goals[0], 'goal_index': 0},
        ],
        'agent_1': [
            {'prio': 1, 'name': 'input_limits'},
            # {'prio': 2, 'name': 'input_smooth'},
            # {'prio': 3, 'name': 'obstacle_avoidance'},
            {'prio': 4, 'name': 'formation', 'agents': [[1, 0]], 'distance': 1.5},
            {'prio': 4, 'name': 'formation', 'agents': [[1, 2]], 'distance': 1.5},
            {'prio': 4, 'name': 'formation', 'agents': [[1, 4]], 'distance': 2.12},
            # {'prio': 3, 'name': 'position', 'goal': goals[1], 'goal_index': 1},
            # {'prio': 3, 'name': 'collision_avoidance'},
        ],
        'agent_2': [
            {'prio': 1, 'name': 'input_limits'},
            {'prio': 2, 'name': 'input_smooth'},
            # {'prio': 3, 'name': 'obstacle_avoidance'},
            {'prio': 4, 'name': 'formation', 'agents': [[1, 2]], 'distance': 1.5},
            {'prio': 4, 'name': 'formation', 'agents': [[3, 2]], 'distance': 1.5},
            {'prio': 4, 'name': 'formation', 'agents': [[2, 4]], 'distance': 1.06},
            # {'prio': 3, 'name': 'position', 'goal': goals[2], 'goal_index': 2},
        ],
        'agent_3': [
            {'prio': 1, 'name': 'input_limits'},
            {'prio': 2, 'name': 'input_smooth'},
            # {'prio': 3, 'name': 'obstacle_avoidance'},
            {'prio': 4, 'name': 'formation', 'agents': [[3, 4]], 'distance': 1.5},
            {'prio': 4, 'name': 'formation', 'agents': [[3, 2]], 'distance': 1.5},
            {'prio': 4, 'name': 'formation', 'agents': [[3, 0]], 'distance': 1.06},
        ],
        # 'agent_4': [
        #     {'prio': 1, 'name': 'input_limits'},
        #     {'prio': 2, 'name': 'input_smooth'},
        #     #{'prio': 3, 'name': 'obstacle_avoidance'},
        #     {'prio': 4, 'name': 'formation', 'agents': [[4, 0]], 'distance': 1.5},
        #     {'prio': 4, 'name': 'formation', 'agents': [[4, 3]], 'distance': 1.5},
        #     {'prio': 4, 'name': 'formation', 'agents': [[4, 5]], 'distance': 1.06}
        # ],
        'agent_4': [
            {'prio': 1, 'name': 'input_limits'},
            {'prio': 2, 'name': 'input_smooth'},
            {'prio': 2, 'name': 'obstacle_avoidance'},
            {'prio': 4, 'name': 'formation', 'agents': [[1, 4]], 'distance': 1.06},
            {'prio': 4, 'name': 'formation', 'agents': [[2, 4]], 'distance': 1.06},
            {'prio': 4, 'name': 'formation', 'agents': [[0, 4]], 'distance': 1.06},
            {'prio': 4, 'name': 'formation', 'agents': [[3, 4]], 'distance': 1.06},
            {'prio': 3, 'name': 'position', 'goal': goals[0], 'goal_index': 0},
        ],
    }
    """system_tasks = {
        'agent_0': [
                {'prio': 1, 'name': 'input_limits'},
                {'prio': 2, 'name': 'input_smooth'},
                #{'prio': 3, 'name': 'obstacle_avoidance'},
                {'prio': 4, 'name': 'formation', 'agents': [[0, 1]], 'distance': 2.5},
                {'prio': 4, 'name': 'formation', 'agents': [[0, 2]], 'distance': 5},
                #{'prio': 3, 'name': 'formation', 'agents': [[0, 2]], 'distance': 4},
                # {'prio': 4, 'name': 'position', 'goal': goals[0], 'goal_index': 0},
            ],
        'agent_1': [
            {'prio': 1, 'name': 'input_limits'},
            {'prio': 2, 'name': 'input_smooth'},
            #{'prio': 3, 'name': 'obstacle_avoidance'},
            {'prio': 4, 'name': 'formation', 'agents': [[1, 0]], 'distance': 2.5},
            {'prio': 4, 'name': 'formation', 'agents': [[1, 2]], 'distance': 2.5},
            {'prio': 3, 'name': 'position', 'goal': goals[0], 'goal_index': 0},
            # {'prio': 3, 'name': 'collision_avoidance'},
        ],
        'agent_2': [
            {'prio': 1, 'name': 'input_limits'},
            {'prio': 2, 'name': 'input_smooth'},
            #{'prio': 3, 'name': 'obstacle_avoidance'},
            {'prio': 4, 'name': 'formation', 'agents': [[0, 2]], 'distance': 5},
            {'prio': 4, 'name': 'formation', 'agents': [[1, 2]], 'distance': 2.5},
            #{'prio': 3, 'name': 'position', 'goal': goals[2], 'goal_index': 2},
        ],
    }"""
elif experiment_name == 'obst_avoid':
    n_nodes = 2
    goals = [
        np.array([-1.5, -0.2]),
        np.array([1.6, -0.1]),
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
            {'prio': 3, 'name': 'obstacle_avoidance'},
            {'prio': 4, 'name': 'position', 'goal': goals[1], 'goal_index': 1},
        ],
        'agent_2': [
            {'prio': 1, 'name': 'input_limits'},
            {'prio': 2, 'name': 'input_smooth'},
            {'prio': 3, 'name': 'obstacles_avoidance'},
            {'prio': 4, 'name': 'formation', 'agents': [[2, 0]], 'distance': 4},
            {'prio': 4, 'name': 'formation', 'agents': [[2, 1]], 'distance': 5.65},
            {'prio': 4, 'name': 'formation', 'agents': [[2, 3]], 'distance': 4},
            # {'prio': 3, 'name': 'position', 'goal': goals[2], 'goal_index': 2},
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
n_steps = 1400
inner_loop = 1  # number of inner loop of the distributed algorithm

communication_range = 8

v_max = 0.15
v_min = 0.0
omega_max = 0.45
omega_min = -0.45

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
