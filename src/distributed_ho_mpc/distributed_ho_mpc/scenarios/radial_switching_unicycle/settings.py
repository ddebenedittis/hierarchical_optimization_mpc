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
# comparison harness overwrites this per run (see methods/dhqp_adapter.py) so
# the enforced radius is an explicit, recorded parameter rather than a constant
# buried in node.py.
safety_distance = 2.0

# ---------------------------------------------------------------------------- #
#                              Scenario definition                             #
# ---------------------------------------------------------------------------- #
# 'uniform':           every agent runs the same task hierarchy.
# 'priority_conflict': agents carry DIFFERENT local hierarchies, including two
#                      that rank the same shared formation task in opposite
#                      order. Reactive baselines (ORCA, CBF-QP) cannot express
#                      this at all -- they have no notion of a task hierarchy,
#                      let alone a per-agent one.
scenario = 'uniform'

# (agent_a, agent_b, target center-to-center distance) coupled by a formation
# task. Only instantiated when scenario == 'priority_conflict'.
formation_pairs = [(0, 1, 3.0)]

# Per-agent priority overrides layered on DEFAULT_TASK_PRIORITIES in
# network_simulation.py. Agent a ranks its own goal above the shared formation;
# agent b ranks the formation above its own goal, so the pair disagrees about
# the same task and the hierarchy -- not a weight -- decides which one yields.
#
# Every hierarchy here must use at most `n_priority` distinct levels: y_i/rho_i
# in node.py are sized by it, so a 5th level indexes out of bounds. That is why
# collision_avoidance shares level 2 with input_smooth rather than taking its
# own, freeing levels 3 and 4 for the position/formation swap.
priority_overrides = {
    0: {'input_smooth': 2, 'collision_avoidance': 2, 'position': 3, 'formation': 4},
    1: {'input_smooth': 2, 'collision_avoidance': 2, 'formation': 3, 'position': 4},
}
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
