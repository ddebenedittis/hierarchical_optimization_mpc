import numpy as np

# ---------------------------------------------------------------------------- #
#                               Network settings                               #
# ---------------------------------------------------------------------------- #
n_nodes = 5  # number of agents
dt = 0.02
n_steps = 3000

v_max = 0.5  # max commanded speed (box constraint on vx, vy)

n_control = 1  # mpc control step
n_pred = 0  # mpc prediction step

# ---------------------------------------------------------------------------- #
#                      Radial-switching start/goal layout                      #
# ---------------------------------------------------------------------------- #
# Same benchmark shape as the other radial-switching scenarios: agents start
# evenly spaced on a circle and must reach the antipodal point.
radius = 4.0

d_safe = 1.0  # inter-robot safety distance (matches the other scenarios)
d_form = 2.0  # desired inter-robot distance for a formation pair
goal_tol = 1e-2  # stop tolerance (matches the paper's own metric definition)

# ---------------------------------------------------------------------------- #
#                                   Scenario                                   #
# ---------------------------------------------------------------------------- #
# 'uniform':
#   Every agent only has a goal-reaching task, protected by input-limit and
#   collision-avoidance safety constraints.
#
# 'priority_conflict':
#   Agents 0 and 1 must additionally hold a formation. Unlike dHQP -- where
#   each agent solves its OWN local hierarchy and can therefore give the
#   pair a genuinely different priority order per agent -- a centralized
#   solver has a single, global task hierarchy shared by the whole fleet.
#   The only well-posed way to express "formation vs. goal conflict" here is
#   to prioritize the formation task above ALL individual goal tasks
#   globally (agents 2-4 are unaffected, since they have no formation task).
#   This is a structural limitation worth reporting on its own: a centralized
#   hierarchy cannot reproduce dHQP's per-agent priority reordering at all.
# 'asymmetric':
#   Same task set as 'uniform' (no formation), but start angles are drawn
#   randomly on the circle instead of evenly spaced (goal is still the
#   antipodal point of wherever the agent actually starts).
scenario = 'priority_conflict'  # 'uniform', 'asymmetric', or 'priority_conflict'

formation_pairs = [(0, 1, d_form)]  # (agent_a, agent_b, target_distance)

min_spawn_distance = 1.5 * d_safe  # used only when scenario == 'asymmetric'

# ---------------------------------------------------------------------------- #
#                              Flags for simulation                            #
# ---------------------------------------------------------------------------- #
output = {'display': 'plot', 'save': 'save', 'nothing': 'none'}
visual_method = output['save']  # change the key to decide the output visualization
