import numpy as np

# ---------------------------------------------------------------------------- #
#                               Network settings                               #
# ---------------------------------------------------------------------------- #
n_nodes = 7  # number of agents
random_graph = False
I_NN = np.identity(n_nodes, dtype=int)

dt = 0.04
n_steps = 600
inner_loop = 1  # number of inner loop of the distributed algorithm

# Real sensing range: connections form/break dynamically as agents move
# (see `neigh_connection` in `network_simulation.py`), same idea as the
# other baselines' per-step `communication_range` filter, but here it
# actually adds/removes tasks via `Node.create_connection`/`remove_connection`
# instead of just gating a force term.
communication_range = 3
neighbor_limit = 3  # connect to at most this many closest agents within range

v_max = 1.0
v_min = -1.0
omega_max = 0.5
omega_min = -0.5

# ---------------------------------------------------------------------------- #
#                      Radial-switching start/goal layout                      #
# ---------------------------------------------------------------------------- #
radius = 7.0

d_form = 2.0  # desired inter-robot distance for a formation pair
goal_tol = 1e-2  # stop tolerance (matches the paper's own metric definition)
d_safe = 2  # inter-robot spawn distance for initial positions and goals
# ---------------------------------------------------------------------------- #
#                                   Scenario                                   #
# ---------------------------------------------------------------------------- #
# 'uniform':
#   Every agent only has a goal-reaching task, protected by input-limit and
#   collision-avoidance tasks.
#
# 'priority_conflict':
#   Agents 0 and 1 must additionally hold a formation with each other. Each
#   dHQP agent solves its OWN local task hierarchy, so the priority order of
#   "formation" vs. "goal" can be set independently per agent: agent 0 puts
#   formation above its own goal, agent 1 puts its own goal above formation
#   -- the same "different priority order for the same task" conflict as
#   the other lightweight baselines, but here resolved by two genuinely
#   different, exactly-enforced local hierarchies rather than a weighted
#   blend. No retuning is needed to switch between scenarios: only the prio
#   integers assigned to the 'position' and 'formation' task dicts change
#   (see `network_simulation.py`).
scenario = 'uniform'  # 'uniform' or 'priority_conflict' or 'asymmetric'
goal_placement = 'symmetric'  # 'symmetric' or 'random'
formation_pairs = [(0, 1, d_form)]  # (agent_a, agent_b, target_distance)

# Each agent's local hierarchy is capped at 4 distinct priority levels (see
# the docstring of `build_system_tasks` in `network_simulation.py`): this
# `node.py`'s solver hardcodes a size-5 dual-variable array that overflows
# beyond that. No 'input_smooth' task as a result.

# ---------------------------------------------------------------------------- #
#                              Flags for simulation                            #
# ---------------------------------------------------------------------------- #
simulation = True
output = {'display': 'plot', 'save': 'save', 'nothing': 'none'}
visual_method = output['nothing']  # change the key to decide the output visualization

n_priority = 4
step_size = 1e-5

# ---------------------------------------------------------------------------- #
#                          Node (radial_switching_unicycle) settings           #
# ---------------------------------------------------------------------------- #
# `node.py` here is a copy of `radial_switching_unicycle/node.py` (only the
# internal `settings` import was redirected to this module) -- it supports
# both robot models but always builds whichever one `type` selects.
type = 'omni'  # 'omni' or 'uni' -- this benchmark uses the omnidirectional model
n_control = 3
n_pred = 0
n_xi = n_control * 4  # dimension of primal variables for the omni model

save_data = False  # skip the per-node CSV logging, not needed for the comparison
inner_plot = False
