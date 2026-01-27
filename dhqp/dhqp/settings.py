import numpy as np

# ---------------------------------------------------------------------------- #
#                               Network settings                               #
# ---------------------------------------------------------------------------- #
experiment_name = 'radial_swt'  # 'form' 'obst_avoid' 'radial_swt'

if experiment_name == 'radial_swt':
    # goals = [
    #     np.array([ 2.50472862, -0.44350808]),  # limo_1
    #     np.array([ 2.60031723,  1.42929871]),  # limo_2
    #     np.array([ 0.30474463,  1.53108807]),  # limo_3
    #     np.array([-0.42369137,  0.22340126]),  # limo_4
    #     np.array([ 0.72717752, -1.07773726])   # limo_5
    # ]
    goals = [
        np.array([-0.21941797, 1.10852516]),  # limo_1
        np.array([-0.31500658, -0.76428163]),  # limo_2
        np.array([1.98056602, -0.86607099]),  # limo_3
        np.array([2.70900202, 0.44161582]),  # limo_4
        np.array([1.55813313, 1.74275434]),  # limo_5
    ]

    moving_obstacle = False

    n_nodes = 5
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
            {'prio': 4, 'name': 'position', 'goal': goals[2], 'goal_index': 2},
        ],
        'agent_3': [
            {'prio': 1, 'name': 'input_limits'},
            {'prio': 2, 'name': 'input_smooth'},
            {'prio': 3, 'name': 'collision_avoidance'},
            {'prio': 4, 'name': 'position', 'goal': goals[3], 'goal_index': 3},
        ],
        'agent_4': [
            {'prio': 1, 'name': 'input_limits'},
            {'prio': 2, 'name': 'input_smooth'},
            {'prio': 3, 'name': 'collision_avoidance'},
            {'prio': 4, 'name': 'position', 'goal': goals[4], 'goal_index': 4},
        ],
        'agent_5': [
            {'prio': 1, 'name': 'input_limits'},
            {'prio': 2, 'name': 'input_smooth'},
            {'prio': 3, 'name': 'collision_avoidance'},
            {'prio': 4, 'name': 'position', 'goal': goals[4], 'goal_index': 4},
        ],
    }
    """system_tasks = {
        'agent_0': [
            {'prio': 1, 'name': 'input_limits'},
            {'prio': 2, 'name': 'input_smooth'},
            {'prio': 3, 'name': 'collision_avoidance'},
            {'prio': 4, 'name': 'coverage'},
        ],
        'agent_1': [
            {'prio': 1, 'name': 'input_limits'},
            {'prio': 2, 'name': 'input_smooth'},
            {'prio': 3, 'name': 'collision_avoidance'},
            {'prio': 4, 'name': 'coverage'},
        ],
        'agent_2': [
            {'prio': 1, 'name': 'input_limits'},
            {'prio': 2, 'name': 'input_smooth'},
            {'prio': 3, 'name': 'collision_avoidance'},
            {'prio': 4, 'name': 'coverage'},
        ],
        'agent_3': [
            {'prio': 1, 'name': 'input_limits'},
            {'prio': 2, 'name': 'input_smooth'},
            {'prio': 3, 'name': 'collision_avoidance'},
            {'prio': 4, 'name': 'coverage'},
        ],
    }"""
elif experiment_name == 'form':
    n_nodes = 5
    goals = [
        np.array([0, 0]),
    ]
    moving_obstacle = True
    system_tasks = {
        'agent_0': [
            {'prio': 1, 'name': 'input_limits'},
            {'prio': 2, 'name': 'input_smooth'},
            {'prio': 3, 'name': 'collision_avoidance'},
            {'prio': 3, 'name': 'obstacle_avoidance'},
            {'prio': 4, 'name': 'coverage'},
        ],
        'agent_1': [
            {'prio': 1, 'name': 'input_limits'},
            {'prio': 2, 'name': 'input_smooth'},
            {'prio': 3, 'name': 'collision_avoidance'},
            {'prio': 3, 'name': 'obstacle_avoidance'},
            {'prio': 4, 'name': 'coverage'},
        ],
        'agent_2': [
            {'prio': 1, 'name': 'input_limits'},
            {'prio': 2, 'name': 'input_smooth'},
            {'prio': 3, 'name': 'collision_avoidance'},
            {'prio': 3, 'name': 'obstacle_avoidance'},
            {'prio': 4, 'name': 'coverage'},
        ],
        'agent_3': [
            {'prio': 1, 'name': 'input_limits'},
            {'prio': 2, 'name': 'input_smooth'},
            {'prio': 3, 'name': 'collision_avoidance'},
            {'prio': 3, 'name': 'obstacle_avoidance'},
            {'prio': 4, 'name': 'coverage'},
        ],
        'agent_4': [
            {'prio': 1, 'name': 'input_limits'},
            {'prio': 2, 'name': 'input_smooth'},
            {'prio': 3, 'name': 'collision_avoidance'},
            {'prio': 3, 'name': 'obstacle_avoidance'},
            {'prio': 4, 'name': 'coverage'},
        ],
    }
    """system_tasks = {
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
    }"""
elif experiment_name == 'obst_avoid':
    n_nodes = 4
    moving_obstacle = False
    goals = [
        np.array([3.48, 1.25]),  # np.array([-3.437, -3.618]),
        np.array([3.4, -0.36]),
        np.array([-0.58, -0.59]),
        np.array([-0.61, 1.32]),
        # np.array([-0.58, -0.59]),  # np.array([-3.437, -3.618]),
        # np.array([-0.61, 1.32]),
        # np.array([3.48, 1.25]),
        # np.array([3.4, -0.36]),
    ]
    system_tasks = {
        'agent_0': [
            {'prio': 1, 'name': 'input_limits'},
            {'prio': 2, 'name': 'input_smooth'},
            {'prio': 2, 'name': 'obstacle_avoidance'},
            {'prio': 3, 'name': 'collision_avoidance'},
            {'prio': 4, 'name': 'position', 'goal': goals[0], 'goal_index': 0},
        ],
        'agent_1': [
            {'prio': 1, 'name': 'input_limits'},
            {'prio': 2, 'name': 'input_smooth'},
            {'prio': 2, 'name': 'obstacle_avoidance'},
            {'prio': 3, 'name': 'collision_avoidance'},
            {'prio': 4, 'name': 'position', 'goal': goals[1], 'goal_index': 1},
        ],
        'agent_2': [
            {'prio': 1, 'name': 'input_limits'},
            {'prio': 2, 'name': 'input_smooth'},
            {'prio': 2, 'name': 'obstacle_avoidance'},
            {'prio': 3, 'name': 'collision_avoidance'},
            {'prio': 4, 'name': 'position', 'goal': goals[2], 'goal_index': 2},
        ],
        'agent_3': [
            {'prio': 1, 'name': 'input_limits'},
            {'prio': 2, 'name': 'input_smooth'},
            {'prio': 2, 'name': 'obstacle_avoidance'},
            {'prio': 3, 'name': 'collision_avoidance'},
            {'prio': 4, 'name': 'position', 'goal': goals[3], 'goal_index': 3},
        ],
    }
p = 1  # probability of arc of communication
# n_nodes = 1  # numbers of nodes MOVED INSIDE IF DEPENDING ON EXPERIMENT
random_graph = False  # create a random graph or not
I_NN = np.identity(n_nodes, dtype=int)
R = 0.045
L = 0.173
dt = 0.05
n_steps = 180
inner_loop = 1  # number of inner loop of the distributed algorithm

communication_range = 1

v_max = 0.2
v_min = 0.0
omega_max = 0.4
omega_min = -0.4
velocity_limits = [v_min, v_max, omega_min, omega_max]
bounding_box = [-2.5, 2.5, -2.5, 2.5]  # xmin, xmax, ymin, ymax


# ---------------------------------------------------------------------------- #
#                              Flags for simulation                            #
# ---------------------------------------------------------------------------- #
output = {'display': 'plot', 'save': 'save', 'nothing': 'none'}
visual_method = output['save']  # change the key to decide the output visualization
save_data = True
simulation = True
inner_plot = False  # plot the inner state of the robots
estimation_plotting = False
# ---------------------------------------------------------------------------- #
#                                 MPC settings                                 #
# ---------------------------------------------------------------------------- #
n_control = 4  # mpc control step
n_pred = 0  # mpc prediction step

n_xi = n_control * 5
safety_distance = 0.6  # minimum safety distance between robots
obstacle_position = np.array([0.5, 0])
obstacle_size = 0.4
vel = np.array([-0.05, 0.0])
# --------------------------------------------------------------------------- #
#                                 PDD settings                               #
# ---------------------------------------------------------------------------- #


n_priority = 4
step_size = 1e-5


variable_connection = False
n_connection = 2  # number of maximum connections per node
