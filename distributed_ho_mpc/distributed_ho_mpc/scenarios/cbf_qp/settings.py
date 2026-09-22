import numpy as np

from hierarchical_qp.hierarchical_qp import QPSolver

# ---------------------------------------------------------------------------- #
#                               Network settings                               #
# ---------------------------------------------------------------------------- #
n_nodes = 5  # number of agents
dt = 0.1
n_steps = 400

communication_range = 8  # max sensing range for the collision/formation constraints

v_max = 0.5  # max commanded speed

# ---------------------------------------------------------------------------- #
#                      Radial-switching start/goal layout                      #
# ---------------------------------------------------------------------------- #
# Agents start evenly spaced on a circle and must reach the antipodal point,
# so every pair of trajectories crosses near the centre. Same benchmark shape
# as the dHQP radial-switching and potential-field scenarios.
radius = 4.0

# ---------------------------------------------------------------------------- #
#                        Distributed-QP task parameters                        #
# ---------------------------------------------------------------------------- #
solver = QPSolver.quadprog  # {clarabel, osqp, proxqp, quadprog, reluqp}

k_goal = 0.6  # nominal goal-tracking gain (equality task RHS)
d_safe = 1.0  # measurement threshold (matches dHQP/potential-field/comparison's shared d_safe)
l = 0.15  # extra conservative buffer this method holds beyond d_safe (agent footprint/inflation)
# The CBF barrier enforces d_safe_enforced = d_safe + 2*l, not d_safe directly -- this baseline
# is deliberately more conservative than the shared measurement threshold, the same way a real
# CBF-QP controller needs margin beyond the metric's collision distance to stay feasible/robust.
gamma = 4.0  # class-K CBF gain: how aggressively the safety margin may be recovered

w_safety = 1e3  # heavily-penalized SOFT weight for velocity-limit + CBF constraints (not
# np.inf): satisfied to within ~1e-4 of the target margin whenever the exact hard problem
# would be feasible, but degrades gracefully instead of raising RuntimeError when several
# neighbors are simultaneously pinned at the enforced margin in conflicting directions and
# the hard problem has no solution at all -- see node.py's docstring and README.md.
# Keep this well below ~1e4: with wi this large, the slack block's diagonal (wi**2) becomes
# ~1e8 vs. ~1 for the goal-tracking block, and `quadprog` starts returning `None` (spurious
# "infeasible") for a well-posed, positive-definite QP purely from that conditioning gap --
# confirmed empirically (1e2/1e3/3e3 all run clean through a full 1200-step campaign with
# min_distance matching the enforced margin to ~1e-4; 1e4 fails partway through).

k_form = 1.5  # formation-keeping gain (equality task RHS)
d_form = 2.0  # desired inter-robot distance for a formation pair (matches dHQP/potential-field)

# Relative blend weights for the two soft equality costs (goal vs. formation).
# Both tasks are solved TOGETHER in one non-hierarchical QP (`hierarchical=False`);
# there is no priority order between them -- only the ratio of these weights
# decides the trade-off when the two disagree. Velocity limits and collision
# avoidance are always hard (see `node.py`), regardless of these weights.
w_goal = 1.0
w_form = 1.0

goal_tol = 1e-2  # stop tolerance

# ---------------------------------------------------------------------------- #
#                                   Scenario                                   #
# ---------------------------------------------------------------------------- #
# 'uniform':
#   Every agent only has a goal-reaching task, protected by the (always
#   hard) velocity-limit and collision-avoidance safety constraints -- same
#   regime as the potential-field baseline's 'uniform'.
#
# 'priority_conflict':
#   Agents 0 and 1 must also hold a formation with each other. Both the
#   goal-reaching and formation-keeping tasks are equality costs solved in
#   the SAME single QP, weighted by `w_goal`/`w_form` -- exactly the same
#   kind of soft trade-off the potential-field baseline makes with its
#   `k_goal`/`k_form` gains, just resolved by an actual weighted-least-squares
#   QP solve instead of a raw heuristic vector sum, and with hard safety
#   constraints guaranteed regardless of the weight choice. Agent 0 is
#   weighted to favour formation; agent 1 is weighted to favour its own goal.
# 'asymmetric':
#   Same task set as 'uniform' (no formation), but start angles are drawn
#   randomly on the circle instead of evenly spaced (goal is still the
#   antipodal point of wherever the agent actually starts).
scenario = 'priority_conflict'  # 'uniform', 'asymmetric', or 'priority_conflict'

formation_pairs = [(0, 1, d_form)]  # (agent_a, agent_b, target_distance)

min_spawn_distance = 1.5 * d_safe  # used only when scenario == 'asymmetric'

# Per-agent overrides of (w_goal, w_form) used when scenario == 'priority_conflict'.
weight_overrides = {
    0: {'w_goal': 0.2, 'w_form': 4.0},  # agent 0: formation dominates its own goal
    1: {'w_goal': 4.0, 'w_form': 0.3},  # agent 1: its own goal dominates formation
}

# ---------------------------------------------------------------------------- #
#                              Flags for simulation                            #
# ---------------------------------------------------------------------------- #
output = {'display': 'plot', 'save': 'save', 'nothing': 'none'}
visual_method = output['display']  # change the key to decide the output visualization
