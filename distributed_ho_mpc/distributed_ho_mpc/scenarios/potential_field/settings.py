import numpy as np

# ---------------------------------------------------------------------------- #
#                               Network settings                               #
# ---------------------------------------------------------------------------- #
n_nodes = 5  # number of agents
dt = 0.1
n_steps = 400

communication_range = 8  # max sensing range for repulsion/formation forces

v_max = 0.5  # max commanded speed

# ---------------------------------------------------------------------------- #
#                      Radial-switching start/goal layout                      #
# ---------------------------------------------------------------------------- #
# Agents start evenly spaced on a circle and must reach the antipodal point,
# so every pair of trajectories crosses near the centre. Same benchmark shape
# as the dHQP radial-switching scenario, but generated procedurally.
radius = 4.0

# ---------------------------------------------------------------------------- #
#                       Potential-field / behaviour gains                      #
# ---------------------------------------------------------------------------- #
k_goal = 0.6  # attraction gain towards the individual goal
k_rep = 3.0  # repulsion gain for collision avoidance
d_safe = 1.0  # inter-robot distance below which repulsion activates (matches dHQP threshold)
k_form = 1.5  # spring gain for the formation-keeping behaviour
d_form = 2.0  # desired inter-robot distance for a formation pair (matches dHQP)

goal_tol = 1e-2  # stop tolerance

# ---------------------------------------------------------------------------- #
#                                   Scenario                                   #
# ---------------------------------------------------------------------------- #
# 'uniform':
#   Every agent only seeks its own goal while avoiding collisions. This is the
#   case a lightweight reactive controller is designed for, and where it is
#   expected to be competitive with (and cheaper than) dHQP.
#
# 'priority_conflict':
#   Agents 0 and 1 must additionally hold a formation with each other while
#   still radially switching. A potential-field/behaviour-based controller can
#   only blend objectives through fixed scalar weights -- it has no notion of
#   a strict priority order. Giving the two agents in the pair a different
#   relative weighting of "hold formation" vs. "reach my own goal" (exactly
#   what "different priority order for the same task" means for dHQP) turns
#   into a tug-of-war with no gain choice that satisfies both agents' intent
#   exactly, whereas dHQP handles this by simply reordering the task hierarchy
#   per agent, with no retuning.
scenario = 'priority_conflict'  # 'uniform' or 'priority_conflict'

formation_pairs = [(0, 1, d_form)]  # (agent_a, agent_b, target_distance)

priority_overrides = {
    0: {'k_goal': 0.15, 'k_form': 4.0},  # agent 0: formation dominates its own goal
    1: {'k_goal': 1.2, 'k_form': 0.3},  # agent 1: its own goal dominates the formation
    2: {'k_goal': 4.0, 'k_form': 0.0},
}

# ---------------------------------------------------------------------------- #
#                              Flags for simulation                            #
# ---------------------------------------------------------------------------- #
output = {'display': 'plot', 'save': 'save', 'nothing': 'none'}
visual_method = output['display']  # change the key to decide the output visualization
