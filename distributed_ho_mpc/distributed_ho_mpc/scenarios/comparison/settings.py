"""Canonical benchmark parameters shared by every method in the comparison.

Every scenario folder under `distributed_ho_mpc/scenarios/` defines its own
`settings.py` with its own defaults (useful for developing/tuning that
method in isolation). `run_comparison.py` monkey-patches each method's
settings module to these canonical values before calling its `run()`, so
all methods are evaluated on the literal same benchmark instance.
"""

n_nodes = 6
radius = 6.0  # circle radius for the radial start/goal layout

dt = 0.03
n_steps = 1200  # 60 s cap -- generous enough for every method to converge

communication_range = 4  # > 2 * radius: every agent senses every other agent
neighbor_limit = (
    2  # limit the number of neighbors to this many closest agents if possible (so in dhqp)
)

v_max = 1.4
v_min = -1.0
d_safe = 1.5  # inter-robot safety distance
d_form = 2.0  # desired inter-robot distance for the formation pair
n_control = 2  # mpc control step
n_pred = 0  # mpc prediction step

goal_tol = 1e-2  # matches the paper's own "normalized time-to-goal" definition (1 cm)
form_tol = 2e-2


formation_pairs = [(0, 1, d_form)]  # (agent_a, agent_b, target_distance)

# Minimum spawn separation enforced when a scenario uses a random layout
# ('asymmetric'). Rejection-sampled against the circle geometry -- see
# `build_radial_configuration(..., layout='random', ...)` in each method's
# `network_simulation.py`.
min_spawn_distance = 1.5 * d_safe

scenarios = ['uniform', 'asymmetric', 'priority_conflict']

output_dir_name = 'comparison'
