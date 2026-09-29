import numpy as np

# ---------------------------------------------------------------------------- #
#                               Network settings                               #
# ---------------------------------------------------------------------------- #
n_nodes = 5  # number of agents
dt = 0.1
n_steps = 200

communication_range = 6  # max sensing range: only neighbours within this range are avoided

v_max = 0.5  # max commanded speed

# ---------------------------------------------------------------------------- #
#                      Radial-switching start/goal layout                      #
# ---------------------------------------------------------------------------- #
# Agents start on a circle and must reach the antipodal point (still on the
# same circle, straight through the centre), so every pair of trajectories
# crosses near the centre. Same benchmark shape as the dHQP radial-switching
# scenario (and potential_field's), generated procedurally.
radius = 4.0

# 'symmetric': agents start evenly spaced on the circle (deterministic) --
#              every trajectory pattern is identical up to rotation, which
#              can hide asymmetric failure modes.
# 'random':    agents start at random angles on the same circle instead,
#              breaking that symmetry while keeping the same "cross through
#              the centre" benchmark shape. Layouts where any two agents
#              would spawn closer than `min_spawn_distance` are rejected and
#              resampled.
layout = 'symmetric'  # 'symmetric' or 'random'
min_spawn_distance = 2.0  # minimum spawn separation enforced when layout == 'random'

# ---------------------------------------------------------------------------- #
#                             Go-to-goal controller                            #
# ---------------------------------------------------------------------------- #
k_goal = 1.0  # proportional gain turning the position error into a preferred velocity
goal_tol = 1e-2  # stop tolerance

# ---------------------------------------------------------------------------- #
#                                 ORCA settings                                #
# ---------------------------------------------------------------------------- #
orca_radius = 0.5  # collision radius of each agent
orca_time_horizon = 2.0  # look-ahead time tau used to build the velocity obstacle
orca_max_speed = v_max  # speed disk used by the ORCA linear program

# ORCA has no notion of a formation task: it always runs the plain
# go-to-goal + collision-avoidance law. This flag is purely informational,
# kept for interface parity with the other scenarios (the comparison
# harness sets it uniformly across all methods before each run).
scenario = 'uniform'  # 'uniform' or 'priority_conflict' -- has no effect on the control law

# ---------------------------------------------------------------------------- #
#                              Flags for simulation                            #
# ---------------------------------------------------------------------------- #
output = {'display': 'plot', 'save': 'save', 'nothing': 'none'}
visual_method = output['save']  # change the key to decide the output visualization
