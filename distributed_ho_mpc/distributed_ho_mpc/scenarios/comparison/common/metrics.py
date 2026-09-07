"""KPI metrics for radial-switching comparison runs.

``count_collisions`` and ``compute_violation_stats`` lift the logic from
``radial_switching_unicycle/network_simulation.py``, generalized to accept
any (T, N, >=2) position/state array (only the first two columns, x and y,
are used).
"""

from __future__ import annotations

import itertools

import numpy as np


def count_collisions(x_hist: np.ndarray, safety_distance: float) -> tuple[int, dict]:
    """Count distinct collision events per robot pair.

    A collision event is a maximal contiguous run of timesteps where a
    pair's distance stays below safety_distance; a single prolonged
    collision counts once, not once per timestep.

    Args:
        x_hist: State/position history, shape (T, N, >=2); only the first
            two columns (x, y) are used.
        safety_distance: Distance threshold defining a collision.

    Returns:
        Tuple (total_events, per_pair_events) where per_pair_events maps
        (i, j) -> event count.
    """
    positions = x_hist[..., :2]
    n_robots = positions.shape[1]
    per_pair_events = {}

    for i, j in itertools.combinations(range(n_robots), 2):
        dist = np.linalg.norm(positions[:, i] - positions[:, j], axis=1)
        in_collision = dist < safety_distance
        events = np.sum(in_collision[1:] & ~in_collision[:-1]) + int(in_collision[0])
        per_pair_events[(i, j)] = int(events)

    total_events = sum(per_pair_events.values())
    return total_events, per_pair_events


def compute_violation_stats(x_hist: np.ndarray, safety_distance: float) -> tuple[float, float]:
    """Compute how deep collisions cut into the safety distance.

    For every robot pair and timestep where distance < safety_distance, the
    violation is (safety_distance - distance).

    Args:
        x_hist: State/position history, shape (T, N, >=2); only the first
            two columns (x, y) are used.
        safety_distance: Distance threshold defining a collision.

    Returns:
        Tuple (max_violation, mean_violation) across all such
        pair-timestep samples; (0.0, 0.0) if there were none.
    """
    positions = x_hist[..., :2]
    n_robots = positions.shape[1]
    violations = []

    for i, j in itertools.combinations(range(n_robots), 2):
        dist = np.linalg.norm(positions[:, i] - positions[:, j], axis=1)
        in_collision = dist < safety_distance
        violations.append(safety_distance - dist[in_collision])

    violations = np.concatenate(violations) if violations else np.array([])
    if violations.size == 0:
        return 0.0, 0.0
    return float(np.max(violations)), float(np.mean(violations))


def _min_pairwise_distance(positions: np.ndarray) -> float:
    """Minimum pairwise Euclidean distance over the whole trajectory."""
    n_robots = positions.shape[1]
    if n_robots < 2:
        return float('nan')
    return min(
        float(np.linalg.norm(positions[:, i] - positions[:, j], axis=1).min())
        for i, j in itertools.combinations(range(n_robots), 2)
    )


def compute_kpis(run_info: dict, arrays: dict) -> dict:
    """Compute a flat dict of KPIs from a loaded run.

    Args:
        run_info: The run_info.json dict, as returned by ``load_run``.
        arrays: The trajectory.npz arrays dict, as returned by ``load_run``
            (x_hist, u_hist, solve_times, s_init, goals).

    Returns:
        Flat dict with keys: method, seed, param_<name> for every recorded
        param, collision_events, min_distance, max_violation,
        mean_violation, n_reached, success, makespan, time_to_goal_mean,
        path_length_ratio, control_effort, smoothness, solve_time_mean,
        solve_time_max, infeasible_count, sim_steps.
    """
    x_hist = arrays['x_hist']
    u_hist = arrays['u_hist']
    solve_times = arrays.get('solve_times')
    s_init = arrays['s_init']
    goals = arrays['goals']

    config = run_info.get('config', {}) or {}
    dt = run_info.get('dt', config.get('dt', 0.05))
    goal_tol = config.get('goal_tol', 0.1)
    safety_distance = config.get('safety_distance', 2.0)

    positions = x_hist[..., :2]
    n_steps_plus_1, n_robots = positions.shape[0], positions.shape[1]

    collision_events, _ = count_collisions(x_hist, safety_distance)
    max_violation, mean_violation = compute_violation_stats(x_hist, safety_distance)
    min_distance = _min_pairwise_distance(positions)

    dist_to_goal = np.linalg.norm(positions - goals[None, :, :], axis=2)
    within = dist_to_goal <= goal_tol

    n_reached = int(np.sum(within[-1]))
    success = bool(n_reached == n_robots)

    # First step after which each robot stays within goal_tol through the end.
    first_reach_stay = np.full(n_robots, np.nan)
    for i in range(n_robots):
        not_within_idxs = np.where(~within[:, i])[0]
        start = (not_within_idxs[-1] + 1) if not_within_idxs.size > 0 else 0
        if start < n_steps_plus_1:
            first_reach_stay[i] = start
    makespan = float(dt * np.max(first_reach_stay)) if success else float('nan')

    # First step (not necessarily permanent) each robot is within goal_tol.
    first_within = np.full(n_robots, np.nan)
    for i in range(n_robots):
        within_idxs = np.where(within[:, i])[0]
        if within_idxs.size > 0:
            first_within[i] = within_idxs[0]
    if np.all(np.isnan(first_within)):
        time_to_goal_mean = float('nan')
    else:
        time_to_goal_mean = float(np.nanmean(first_within) * dt)

    diffs = np.diff(positions, axis=0)
    path_lengths = np.linalg.norm(diffs, axis=2).sum(axis=0)
    straight = np.linalg.norm(goals - s_init[:, :2], axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(straight > 0, path_lengths / straight, np.nan)
    path_length_ratio = float(np.nanmean(ratio))

    control_effort = float(np.sum(u_hist[..., 0] ** 2 + u_hist[..., 1] ** 2) * dt)

    if u_hist.shape[0] > 1:
        smoothness = float(np.mean(np.linalg.norm(np.diff(u_hist, axis=0), axis=2)))
    else:
        smoothness = float('nan')

    meta = run_info.get('meta') or {}
    if solve_times is not None and np.any(solve_times):
        solve_time_mean = float(np.mean(solve_times))
        solve_time_max = float(np.max(solve_times))
    else:
        n_solves = meta.get('n_solves')
        total = meta.get('solve_time_total')
        if total is not None and n_solves:
            solve_time_mean = float(total) / float(n_solves)
        else:
            solve_time_mean = float('nan')
        solve_time_max = float(meta['solve_time_max']) if 'solve_time_max' in meta else float('nan')

    kpis = {
        'method': run_info.get('method'),
        'seed': run_info.get('seed'),
        'collision_events': collision_events,
        'min_distance': min_distance,
        'max_violation': max_violation,
        'mean_violation': mean_violation,
        'n_reached': n_reached,
        'success': success,
        'makespan': makespan,
        'time_to_goal_mean': time_to_goal_mean,
        'path_length_ratio': path_length_ratio,
        'control_effort': control_effort,
        'smoothness': smoothness,
        'solve_time_mean': solve_time_mean,
        'solve_time_max': solve_time_max,
        'infeasible_count': run_info.get('infeasible_count', 0),
        'sim_steps': int(u_hist.shape[0]),
    }

    for name, value in (run_info.get('params') or {}).items():
        kpis[f'param_{name}'] = value

    return kpis
