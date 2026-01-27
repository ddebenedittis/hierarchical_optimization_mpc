import numpy as np

# ---------------------------------------------------------------------------- #
#                               Network settings                               #
# ---------------------------------------------------------------------------- #
experiment_name = 'radial_switch'  # 'radial_switch' 'coverage' 'obst_avoid'

if experiment_name == 'radial_switch':
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
        f'agent_{i}': [
            {'prio': 1, 'name': 'input_limits'},
            {'prio': 2, 'name': 'input_smooth'},
            {'prio': 3, 'name': 'collision_avoidance'},
            {'prio': 4, 'name': 'position', 'goal': goals[i], 'goal_index': i},
        ]
        for i in range(n_nodes)
    }
elif experiment_name == 'coverage':
    n_nodes = 5
    goals = [
        np.array([0, 0]),
    ]
    moving_obstacle = True
    system_tasks = {
        f'agent_{i}': [
            {'prio': 1, 'name': 'input_limits'},
            {'prio': 2, 'name': 'input_smooth'},
            {'prio': 3, 'name': 'collision_avoidance'},
            {'prio': 3, 'name': 'obstacle_avoidance'},
            {'prio': 4, 'name': 'coverage'},
        ]
        for i in range(5)
    }
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
        f'agent_{i}': [
            {'prio': 1, 'name': 'input_limits'},
            {'prio': 2, 'name': 'input_smooth'},
            {'prio': 2, 'name': 'obstacle_avoidance'},
            {'prio': 3, 'name': 'collision_avoidance'},
            {'prio': 4, 'name': 'position', 'goal': goals[i], 'goal_index': i},
        ]
        for i in range(n_nodes)
    }

p = 1  # probability of arc of communication

random_graph = False  # create a random graph or not
I_NN = np.identity(n_nodes, dtype=int)
R = 0.045
L = 0.173
dt = 0.05
dt_mpc_model = 0.75
n_steps = 400
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
visual_method = 'save'  # plot, save, none
save_data = True
simulation = True
inner_plot = False  # plot the inner state of the robots
estimation_plotting = False

# ---------------------------------------------------------------------------- #
#                                 MPC settings                                 #
# ---------------------------------------------------------------------------- #
n_control = 6  # mpc control step
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


variable_connection = True
n_connection = 2  # number of maximum connections per node
