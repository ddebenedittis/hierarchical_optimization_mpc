"""Canonical benchmark parameters shared by every method in the comparison.

Every scenario folder under `distributed_ho_mpc/scenarios/` defines its own
`settings.py` with its own defaults (useful for developing/tuning that
method in isolation). `run_comparison.py` monkey-patches each method's
settings module to these canonical values before calling its `run()`, so
all methods are evaluated on the literal same benchmark instance.
"""

import numpy as np
from scipy.spatial.distance import pdist

n_nodes = 6
radius = 6.0  # circle radius for the radial start/goal layout

dt = 0.03
n_steps = 1200  # 60 s cap -- generous enough for every method to converge

communication_range = 6  # > 2 * radius: every agent senses every other agent
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


def _build_symmetric_layout(n_nodes: int, radius: float, seed: int):
    """Evenly-spaced radial layout ('uniform'/'priority_conflict' scenarios),
    rigidly rotated by a random offset drawn from `seed`. Agents stay exactly
    evenly spaced (same relative geometry as the old fixed layout) -- only
    where the first agent (and hence, since goal = -start, its antipodal
    endpoint) sits on the circumference changes with the seed.
    """
    rng = np.random.default_rng(seed)
    rotation = rng.uniform(0.0, 2 * np.pi)
    thetas = rotation + 2 * np.pi * np.arange(n_nodes) / n_nodes
    starts = [radius * np.array([np.cos(t), np.sin(t)]) for t in thetas]
    goals = [-s for s in starts]
    return starts, goals


def _build_asymmetric_layout(
    n_nodes: int,
    radius: float,
    min_spawn_distance: float,
    seed: int = 1,
    max_attempts: int = 1000,
):
    """Draw the ONE random radial start/goal layout used for the
    'asymmetric' scenario across every method. Uses its own RNG stream
    (`np.random.default_rng`, not the global `np.random` state each
    method's `run()` separately seeds) so it is computed once here,
    independent of call order, and handed to every method identically via
    `run_comparison._apply_canonical_settings` -- rather than letting each
    method redraw its own random layout (which, since methods differ in
    layout algorithm and number of RNG draws before this point, do not
    generally agree even when each seeds `np.random` the same way).
    """
    rng = np.random.default_rng(seed)
    if n_nodes > 1:
        max_feasible = 2 * radius * np.sin(np.pi / n_nodes)
        if min_spawn_distance > max_feasible:
            raise ValueError(
                f'min_spawn_distance={min_spawn_distance} is infeasible for '
                f'{n_nodes} agents on a circle of radius={radius} '
                f'(max possible separation is {max_feasible:.3f})'
            )
    for _ in range(max_attempts):
        thetas = rng.uniform(0.0, 2 * np.pi, n_nodes)
        starts = [radius * np.array([np.cos(t), np.sin(t)]) for t in thetas]
        if n_nodes < 2 or pdist(np.array(starts)).min() >= min_spawn_distance:
            break
    else:
        raise RuntimeError(
            f'Could not find a random layout with min_spawn_distance='
            f'{min_spawn_distance} after {max_attempts} attempts'
        )
    goals = [-s for s in starts]
    return starts, goals


scenarios = ['uniform', 'asymmetric', 'priority_conflict']

output_dir_name = 'comparison'
