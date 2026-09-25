import numpy as np

# ---------------------------------------------------------------------------- #
#                               Network settings                               #
# ---------------------------------------------------------------------------- #
p = 1  # probability of arc of communication
n_nodes = 8  # numbers of nodes
random_graph = False  # create a random graph or not
I_NN = np.identity(n_nodes, dtype=int)

dt = 0.05
n_steps = 900
inner_loop = 1  # number of inner loop of the distributed algorithm

communication_range = 3
limit_connection = 2
v_max = 1.6
v_min = -1.6
omega_max = 2.0
omega_min = -2.0
type = 'uni'  # 'uni'

# Center-to-center distance the collision-avoidance task enforces. The
# comparison harness overwrites this per run (see methods/dwqp_adapter.py) so
# the enforced radius is an explicit, recorded parameter rather than a constant
# buried in node.py -- and so dHQP and its weighted ablation can be given the
# same back-off, which is required for the two to stay comparable.
safety_distance = 2.0
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
if type == 'uni':
    n_xi = n_control * 5
elif type == 'omni':
    n_xi = n_control * 4

# ---------------------------------------------------------------------------- #
#                                 PDD settings                               #
# ---------------------------------------------------------------------------- #


n_priority = 4
step_size = 1e-7

kappa = 100.0  # weight ladder base for the weighted (non-hierarchical) solve
hierarchical = False
