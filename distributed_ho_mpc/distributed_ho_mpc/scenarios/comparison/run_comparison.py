"""
Run every distributed-control baseline on the same radial-switching
benchmark (see `settings.py` for the shared parameters) and produce:

  - a per-(method, scenario) KPI table (printed + saved as markdown/CSV)
  - a qualitative capability table (priority hierarchy? per-agent priority
    reordering? formation-type coupling tasks? manual retuning needed?)
  - one overlay plot per scenario of the formation-pair (agents 0-1)
    inter-robot distance over time, for every method on the same axes

Each method's `network_simulation.py` exposes a `run(out_dir, make_plots)`
function returning raw trajectory/timing data (see e.g.
`potential_field/network_simulation.py`); this script monkey-patches each
method's own `settings` module to the canonical values in `settings.py`
before calling it, so every method is evaluated on the literal same
benchmark instance (same start/goal layout, dt, v_max, safety/formation
distances, and stop tolerance).

Run with:
    python3 src/distributed_ho_mpc/distributed_ho_mpc/scenarios/comparison/run_comparison.py
"""

import csv
import importlib
import os
import traceback
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

import distributed_ho_mpc.scenarios.comparison.settings as cst

# (method_key, package path, whether it's expected to run cleanly right now)
METHODS = [
    ('potential_field', 'distributed_ho_mpc.scenarios.potential_field'),
    ('distributed_qp', 'distributed_ho_mpc.scenarios.distributed_qp'),
    ('orca', 'distributed_ho_mpc.scenarios.orca_radial_switching'),
    # ('centralized_hqp', 'distributed_ho_mpc.scenarios.centralized_radial_switching'),
    ('dhqp', 'distributed_ho_mpc.scenarios.dhqp_radial_switching'),
]

# Which canonical attributes exist on each method's own settings module.
ATTR_MAP = {
    'potential_field': [
        'n_nodes',
        'dt',
        'n_steps',
        'communication_range',
        'v_max',
        'radius',
        'd_safe',
        'd_form',
        'goal_tol',
        'min_spawn_distance',
    ],
    'distributed_qp': [
        'n_nodes',
        'dt',
        'n_steps',
        'communication_range',
        'v_max',
        'radius',
        'd_safe',
        'd_form',
        'goal_tol',
        'min_spawn_distance',
    ],
    'orca': [
        'n_nodes',
        'dt',
        'n_steps',
        'communication_range',
        'v_max',
        'radius',
        'goal_tol',
        'min_spawn_distance',
    ],
    'centralized_hqp': [
        'n_nodes',
        'dt',
        'n_steps',
        'v_max',
        'radius',
        'd_safe',
        'd_form',
        'goal_tol',
        'n_control',
        'n_pred',
        'min_spawn_distance',
    ],
    'dhqp': [
        'n_nodes',
        'dt',
        'n_steps',
        'communication_range',
        'neighbor_limit',
        'v_max',
        'v_min',
        'radius',
        'd_form',
        'goal_tol',
        'n_control',
        'n_pred',
    ],
}

# Qualitative facts that cannot be measured by running a simulation --
# whether the method's control law can even express a strict priority
# order, whether that order can differ per agent, and whether changing the
# scenario requires hand-tuning gains. This is the crux of the "flexible
# but not always necessary" argument: it belongs in the table even though
# it isn't a measured quantity.
#
# 'training_cost': cost paid to obtain a working controller for this
#   benchmark, as opposed to 'retuning_to_change_priority' (the cost to
#   adapt an ALREADY-working instance to a different scenario). All 5
#   methods currently here pay zero training cost -- this column exists so
#   a future learned baseline (e.g. GCBF+, see comparison/README.md) has
#   somewhere to report its offline GPU-hours without conflating that with
#   the online retuning cost.
# 'safety_guarantee_type': what kind of safety claim the min-distance/
#   safety-margin measurement actually backs. A learned method's "safe on
#   this rollout" and an exact-constraint method's "safe by construction"
#   are not the same claim even when the measured number looks identical --
#   this column is the caption that keeps them from being conflated.
# 'generalizes_to_unseen_n_agents': whether the method still works after
#   changing the fleet size. True for every method here for the SAME
#   reason (nothing is trained, so there is nothing to generalize -- just
#   re-parametrize `n_nodes`), but a learned policy would need this column
#   for a structurally different reason (a permutation-invariant policy
#   evaluated on more agents than it was trained on), so the "why" is kept
#   in the string rather than collapsed into a bare bool.
CAPABILITIES = {
    'potential_field': {
        'strict_priority': False,
        'per_agent_priority': False,
        'formation_tasks': True,
        'retuning_to_change_priority': 'Yes -- hand-tune 2 gains per agent',
        'training_cost': 'N/A -- no training, hand-tuned gains only',
        'safety_guarantee_type': 'none (heuristic force, no formal guarantee)',
        'generalizes_to_unseen_n_agents': 'Yes -- re-parametrized directly, no retraining needed',
    },
    'distributed_qp': {
        'strict_priority': False,
        'per_agent_priority': False,
        'formation_tasks': True,
        'retuning_to_change_priority': 'Yes -- hand-tune 2 weights per agent',
        'training_cost': 'N/A -- no training, hand-tuned weights only',
        'safety_guarantee_type': 'hard constraint (exact, CBF-QP)',
        'generalizes_to_unseen_n_agents': 'Yes -- re-parametrized directly, no retraining needed',
    },
    'orca': {
        'strict_priority': False,
        'per_agent_priority': False,
        'formation_tasks': False,
        'retuning_to_change_priority': 'N/A -- cannot express a formation/coupling task at all',
        'training_cost': 'N/A -- no training, hand-tuned ORCA parameters only',
        'safety_guarantee_type': 'geometric reciprocity (exact, given shared radius)',
        'generalizes_to_unseen_n_agents': 'Yes -- re-parametrized directly, no retraining needed',
    },
    'centralized_hqp': {
        'strict_priority': True,
        'per_agent_priority': False,
        'formation_tasks': True,
        'retuning_to_change_priority': 'No retuning, but only ONE global order for the whole fleet',
        'training_cost': 'N/A -- no training, exact optimization',
        'safety_guarantee_type': 'hard constraint (exact, hierarchical QP)',
        'generalizes_to_unseen_n_agents': 'Yes -- re-parametrized directly, no retraining needed',
    },
    'dhqp': {
        'strict_priority': True,
        'per_agent_priority': True,
        'formation_tasks': True,
        'retuning_to_change_priority': "No -- just reorder each agent's local task-priority integers",
        'training_cost': 'N/A -- no training, exact optimization',
        'safety_guarantee_type': 'hard constraint (exact, hierarchical QP)',
        'generalizes_to_unseen_n_agents': 'Yes -- re-parametrized directly, no retraining needed',
    },
    # Not yet integrated (see comparison/README.md's GCBF+ section) -- kept
    # here as a placeholder so the schema is ready once it is. NOT added to
    # METHODS, so run_all() never tries to run it.
    'gcbf': {
        'strict_priority': False,
        'per_agent_priority': False,
        'formation_tasks': False,
        'retuning_to_change_priority': 'N/A -- cannot express a formation/coupling task at all',
        'training_cost': 'Offline, GPU-hours per policy (TODO: measure once integrated)',
        'safety_guarantee_type': 'learned (empirical, not formally guaranteed)',
        'generalizes_to_unseen_n_agents': (
            'Yes, by design (permutation-invariant GNN policy) -- not yet verified on this benchmark'
        ),
    },
}


def _apply_canonical_settings(settings_module, method_key, scenario):
    for attr in ATTR_MAP[method_key]:
        setattr(settings_module, attr, getattr(cst, attr))
    if hasattr(settings_module, 'scenario'):
        settings_module.scenario = scenario
    if hasattr(settings_module, 'formation_pairs'):
        settings_module.formation_pairs = cst.formation_pairs
    if method_key == 'orca':
        settings_module.orca_radius = cst.d_safe / 2.0
        settings_module.orca_max_speed = cst.v_max


def _converged(
    scenario: str,
    goal_errors: np.ndarray,
    formation_dist: float,
    d_form: float,
    a: int,
    b: int,
    caps: dict,
) -> bool:
    """
    Whether the run "succeeded", judged against each agent's own intended
    objective rather than uniformly requiring every agent to reach its own
    goal.

    'uniform'/'asymmetric', or 'priority_conflict' for a method with no
    formation task at all (ORCA): everyone's only objective is their own
    goal -- plain "all goal errors below tolerance".

    'priority_conflict' for a formation-capable method: agents outside the
    pair still need their own goal; the pair itself is judged only on the
    formation distance, not on either member's individual goal error.

    Deliberately NOT checking "does the goal-dominant member of the pair
    reach its own goal": which of `a`/`b` ends up goal-dominant vs.
    formation-dominant is a per-method, per-file construction detail (e.g.
    `dhqp_radial_switching`'s hardcoded `system_tasks` currently makes
    agent 1 formation-dominant and agent 0 goal-dominant -- the opposite of
    what `formation_pairs = [(0, 1, d_form)]`'s ordering would suggest), so
    hardcoding "index `b` must reach goal" is fragile and was wrong for
    dHQP specifically: it required the wrong agent's goal and could never
    pass, even on runs that had clearly converged (verified against an
    actual dHQP snapshot where 5/6 agents sit exactly on their goal markers
    and the 6th holds formation instead). `goal_error_agent_a_m` /
    `goal_error_agent_b_m` in the KPI table still report each member's
    individual goal error for the reader to see who "won" -- that's a
    descriptive finding, not something the boolean `converged` flag should
    gatekeep.
    """
    others = [i for i in range(len(goal_errors)) if i not in (a, b)]
    others_ok = bool(np.all(goal_errors[others] < cst.goal_tol)) if others else True

    if scenario != 'priority_conflict' or not caps['formation_tasks']:
        return others_ok and bool(np.all(goal_errors < cst.goal_tol))

    formation_ok = abs(formation_dist - d_form) < cst.form_tol
    return others_ok and formation_ok


def _run_one(method_key: str, module_path: str, scenario: str):
    settings_module = importlib.import_module(f'{module_path}.settings')
    _apply_canonical_settings(settings_module, method_key, scenario)

    sim_module = importlib.import_module(f'{module_path}.network_simulation')
    return sim_module.run(out_dir=None, make_plots=False)


def _compute_kpis(method_key: str, scenario: str, result: dict) -> dict:
    s_history = result['s_history']
    goals = np.array(result['goals'])
    dt = result['dt']
    last_step = result['last_step']

    final_positions = np.array(s_history[-1][1])
    goal_errors = np.linalg.norm(final_positions - goals, axis=1)

    a, b, d_form = cst.formation_pairs[0]
    formation_dist = float(np.linalg.norm(final_positions[a] - final_positions[b]))

    caps = CAPABILITIES[method_key]
    converged = _converged(scenario, goal_errors, formation_dist, d_form, a, b, caps)

    return {
        'method': method_key,
        'scenario': scenario,
        'status': 'ok',
        'converged': converged,
        'time_to_goal_s': round(last_step * dt, 2) if converged else None,
        'normalized_time_to_goal': round(last_step / cst.n_steps, 3) if converged else None,
        'min_distance_m': round(float(result['min_distance']), 3),
        'safety_margin_ok': bool(result['min_distance'] >= cst.d_safe - 1e-6),
        'formation_pair_distance_m': round(formation_dist, 3),
        'formation_error_m': round(abs(formation_dist - d_form), 3),
        'goal_error_agent_a_m': round(float(goal_errors[a]), 3),
        'goal_error_agent_b_m': round(float(goal_errors[b]), 3),
        'wall_time_s': round(result['wall_time_s'], 4),
        'solve_time_s': round(result['solve_time_s'], 4)
        if result['solve_time_s'] is not None
        else None,
        'strict_priority': caps['strict_priority'],
        'per_agent_priority': caps['per_agent_priority'],
        'formation_tasks': caps['formation_tasks'],
        'retuning_to_change_priority': caps['retuning_to_change_priority'],
        'training_cost': caps['training_cost'],
        'safety_guarantee_type': caps['safety_guarantee_type'],
        'generalizes_to_unseen_n_agents': caps['generalizes_to_unseen_n_agents'],
    }


def _pairwise_history(s_history: list, a: int, b: int, dt: float):
    t = np.arange(1, len(s_history) + 1) * dt
    dist = [np.linalg.norm(np.array(frame[1][a]) - np.array(frame[1][b])) for frame in s_history]
    return t, np.array(dist)


def _write_table(rows: list[dict], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    fieldnames = list(rows[0].keys()) if rows else []
    with open(f'{out_dir}/kpi_table.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    header = fieldnames
    lines = ['| ' + ' | '.join(header) + ' |', '| ' + ' | '.join(['---'] * len(header)) + ' |']
    for row in rows:
        lines.append('| ' + ' | '.join(str(row[h]) for h in header) + ' |')
    md = '\n'.join(lines)

    with open(f'{out_dir}/kpi_table.md', 'w') as f:
        f.write(md)

    print()
    print(md)
    print()
    print(f'KPI table written to {out_dir}/kpi_table.{{csv,md}}')


def run_all(out_dir: str | None = None) -> list[dict]:
    if out_dir is None:
        try:
            from ament_index_python.packages import get_package_share_directory

            workspace_dir = f'{get_package_share_directory("distributed_ho_mpc")}/../../../..'
        except Exception:
            workspace_dir = '.'
        out_dir = f'{workspace_dir}/out/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}-comparison/'
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    raw_results = {}

    for scenario in cst.scenarios:
        for method_key, module_path in METHODS:
            print(f'\n=== Running {method_key} / {scenario} ===')
            try:
                result = _run_one(method_key, module_path, scenario)
                raw_results[(method_key, scenario)] = result
                rows.append(_compute_kpis(method_key, scenario, result))
            except Exception as exc:  # noqa: BLE001 -- keep the whole comparison alive
                traceback.print_exc()
                rows.append(
                    {
                        'method': method_key,
                        'scenario': scenario,
                        'status': f'ERROR: {type(exc).__name__}: {exc}',
                        'converged': None,
                        'time_to_goal_s': None,
                        'normalized_time_to_goal': None,
                        'min_distance_m': None,
                        'safety_margin_ok': None,
                        'formation_pair_distance_m': None,
                        'formation_error_m': None,
                        'goal_error_agent_a_m': None,
                        'goal_error_agent_b_m': None,
                        'wall_time_s': None,
                        'solve_time_s': None,
                        'strict_priority': CAPABILITIES[method_key]['strict_priority'],
                        'per_agent_priority': CAPABILITIES[method_key]['per_agent_priority'],
                        'formation_tasks': CAPABILITIES[method_key]['formation_tasks'],
                        'retuning_to_change_priority': CAPABILITIES[method_key][
                            'retuning_to_change_priority'
                        ],
                        'training_cost': CAPABILITIES[method_key]['training_cost'],
                        'safety_guarantee_type': CAPABILITIES[method_key]['safety_guarantee_type'],
                        'generalizes_to_unseen_n_agents': CAPABILITIES[method_key][
                            'generalizes_to_unseen_n_agents'
                        ],
                    }
                )

    _write_table(rows, out_dir)

    a, b, _ = cst.formation_pairs[0]
    for scenario in cst.scenarios:
        plt.figure(figsize=(9, 5))
        for method_key, _ in METHODS:
            result = raw_results.get((method_key, scenario))
            if result is None:
                continue
            t, dist = _pairwise_history(result['s_history'], a, b, result['dt'])
            plt.plot(t, dist, label=method_key)
        plt.axhline(y=cst.d_safe, color='red', lw=1.5, linestyle='--', label='collision threshold')
        if scenario == 'priority_conflict':
            plt.axhline(
                y=cst.d_form, color='green', lw=1.5, linestyle='--', label='formation target'
            )
        plt.title(f'Agents {a}-{b} distance over time -- {scenario}')
        plt.xlabel('Time [s]')
        plt.ylabel('Distance [m]')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{out_dir}/overlay_{scenario}.pdf', bbox_inches='tight', format='pdf')
        plt.close()

    print(f'\nOverlay plots written to {out_dir}/overlay_<scenario>.pdf')

    return rows


def main():
    run_all()


if __name__ == '__main__':
    main()
