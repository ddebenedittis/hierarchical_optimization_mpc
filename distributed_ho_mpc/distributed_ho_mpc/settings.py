import numpy as np

# ---------------------------------------------------------------------------- #
#                               Network settings                               #
# ---------------------------------------------------------------------------- #
p = 1   # probability of arc of communication
n_nodes = 2 # numbers of nodes
random_graph = True    # create a random graph or not
I_NN = np.identity(n_nodes, dtype=int)
MAX_iter = 40 # Number of iteration 
dt = 0.1
n_steps = 200

formation_distance = 4

output = {
    'display':'plot', 
    'save': 'save', 
    'nothing' :'none'
}
visual_method = output['display'] # change the key to decide the output visualization
# ---------------------------------------------------------------------------- #
#                                 MPC settings                                 #
# ---------------------------------------------------------------------------- #
n_control = 3 # mpc control step
n_pred = 0 # mpc prediction step