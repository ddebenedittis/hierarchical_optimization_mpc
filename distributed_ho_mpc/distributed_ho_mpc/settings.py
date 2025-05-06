import numpy as np

# ---------------------------------------------------------------------------- #
#                               Network settings                               #
# ---------------------------------------------------------------------------- #
p = 1   # probability of arc of communication
n_nodes = 3 # numbers of nodes
random_graph = False    # create a random graph or not
I_NN = np.identity(n_nodes, dtype=int)
MAX_iter = 40 # Number of iteration 
dt = 0.1
n_steps = 150

formation_distance = 4
plot_dual = False

output = {
    'display':'plot', 
    'save': 'save', 
    'nothing' :'none'
}
visual_method = output['display'] # change the key to decide the output visualization

n_priority = 2
# ---------------------------------------------------------------------------- #
#                                 MPC settings                                 #
# ---------------------------------------------------------------------------- #
n_control = 1 # mpc control step
n_pred = 0 # mpc prediction step

n_xi = n_control * 2

step_size = 1e-7