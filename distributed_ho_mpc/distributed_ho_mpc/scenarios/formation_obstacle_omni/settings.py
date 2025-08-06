import numpy as np

# ---------------------------------------------------------------------------- #
#                               Network settings                               #
# ---------------------------------------------------------------------------- #
p = 1  # probability of arc of communication
n_nodes = 9  # numbers of nodes
random_graph = False  # create a random graph or not
I_NN = np.identity(n_nodes, dtype=int)

dt = 0.05
n_steps = 100

communication_range = 9

v_max = 2
v_min = -2

# ---------------------------------------------------------------------------- #
#                              Flags for simulation                            #
# ---------------------------------------------------------------------------- #
output = {'display': 'plot', 'save': 'save', 'nothing': 'none'}
visual_method = output['save']  # change the key to decide the output visualization
save_data = True
simulation = True
estimation_plotting = False

inner_plot = False  # plot the inner state of the robots
# ---------------------------------------------------------------------------- #
#                                 MPC settings                                 #
# ---------------------------------------------------------------------------- #
n_control = 1  # mpc control step
n_pred = 0  # mpc prediction step

n_xi = n_control * 2

# ---------------------------------------------------------------------------- #
#                                 PDD settings                               #
# ---------------------------------------------------------------------------- #

n_priority = 2
step_size = 1e-6
