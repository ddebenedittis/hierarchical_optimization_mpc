"""Adapters running the colleague's omnidirectional baselines on injected instances.

Each ``run_<method>`` takes a ``BenchmarkInstance`` from
``generate_omni_instance``, writes the benchmark values into that method's own
``settings`` module, calls its ``network_simulation.run()`` and repackages the
result as a ``RunResult``. Settings modules are module-level state, so the
campaign runs every job in a fresh process (see ``run_comparison.py``).

The shared rules every method gets from here:

- the same start/goal layout, dt, step cap, v_max, sensing range, goal and
  formation tolerances;
- the same enforced collision distance for every constraint-based method,
  ``d_safe + discretization_back_off(config)``; the potential field has no
  constraint and gets ``d_safe`` as its repulsion radius;
- the same plant saturation ``||u|| <= v_max`` (inside each simulator).

Everything is still scored against ``config.safety_distance``.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np

from distributed_ho_mpc.scenarios.comparison.common.benchmark import (
    BenchmarkInstance,
    discretization_back_off,
)
from distributed_ho_mpc.scenarios.comparison.common.run_io import RunResult

_PKG = 'distributed_ho_mpc.scenarios'

# What each method can express. Used by the metrics to pick the convergence rule
# (a method without formation tasks must bring every agent to its goal).
CAPABILITIES = {
    'pf': {'formation_tasks': True, 'strict_priority': False, 'per_agent_priority': False},
    'dqp': {'formation_tasks': True, 'strict_priority': False, 'per_agent_priority': False},
    'cbf_omni': {'formation_tasks': True, 'strict_priority': False, 'per_agent_priority': False},
    'orca_omni': {'formation_tasks': False, 'strict_priority': False, 'per_agent_priority': False},
    'nh_orca_omni': {
        'formation_tasks': False,
        'strict_priority': False,
        'per_agent_priority': False,
    },
    'dhqp_omni': {'formation_tasks': True, 'strict_priority': True, 'per_agent_priority': True},
}


def enforced_distance(instance: BenchmarkInstance, params: dict) -> tuple[float, float]:
    """Return (enforced distance, back-off) for a constraint-based method."""
    config = instance.config
    back_off = params.get('margin', discretization_back_off(config))
    return config.safety_distance + back_off, back_off


def _load(folder: str):
    st = importlib.import_module(f'{_PKG}.{folder}.settings')
    sim = importlib.import_module(f'{_PKG}.{folder}.network_simulation')
    return st, sim


def _apply_common(st, instance: BenchmarkInstance, params: dict) -> None:
    """Write the benchmark values shared by every method into a settings module."""
    config = instance.config
    st.n_nodes = config.n_robots
    st.dt = config.dt
    st.n_steps = config.max_steps
    st.communication_range = params.get('comm_range', config.comm_range)
    st.v_max = config.v_max
    st.radius = config.radius
    st.goal_tol = config.goal_tol
    st.form_tol = config.form_tol
    st.d_form = config.d_form
    st.scenario = config.scenario
    st.formation_pairs = [tuple(p) for p in config.formation_pairs]
    st.min_spawn_distance = config.min_spawn_distance
    st.fixed_starts = [np.array(p, dtype=float) for p in instance.s_init]
    st.fixed_goals = [np.array(g, dtype=float) for g in instance.goals]
    st.visual_method = 'none'
    if hasattr(st, 'd_safe'):
        st.d_safe = config.safety_distance
    for name, value in params.get('settings', {}).items():
        setattr(st, name, value)


def _to_result(method: str, out: dict, meta: dict) -> RunResult:
    x_hist = np.asarray(out['x_hist'], dtype=float)
    u_hist = np.asarray(out['u_hist'], dtype=float)
    solve_times = np.asarray(out['solve_times'], dtype=float)
    meta = {
        'wall_time_s': float(out['wall_time_s']),
        'solve_time_total': float(np.sum(solve_times)),
        'n_solves': int(solve_times.size),
        'solve_time_max': float(np.max(solve_times)) if solve_times.size else 0.0,
        'infeasible_measured': True,
        'last_step': int(out['last_step']),
        'enforced_safety_distance': (
            None
            if out.get('enforced_safety_distance') is None
            else float(out['enforced_safety_distance'])
        ),
        'capabilities': CAPABILITIES[method],
        **meta,
    }
    arrays = {}
    if out.get('qp_times') is not None:
        arrays['qp_times'] = np.asarray(out['qp_times'], dtype=float)
        meta['qp_time_total'] = float(np.sum(arrays['qp_times']))
    if out.get('link_events') is not None:
        arrays['link_events'] = np.asarray(out['link_events'], dtype=int)
        meta['link_events_total'] = int(np.sum(arrays['link_events']))
    return RunResult(
        x_hist=x_hist,
        u_hist=u_hist,
        solve_times=solve_times,
        infeasible_count=int(out.get('infeasible_count', 0)),
        meta=meta,
        arrays=arrays,
    )


def run_pf(instance: BenchmarkInstance, params: dict, out_dir: Path) -> RunResult:
    """Potential field. No constraint: d_safe is only its repulsion radius."""
    st, sim = _load('potential_field')
    _apply_common(st, instance, params)
    out = sim.run(out_dir=None, make_plots=False)
    out['enforced_safety_distance'] = None
    return _to_result('pf', out, {'back_off': 0.0})


def run_dqp(instance: BenchmarkInstance, params: dict, out_dir: Path) -> RunResult:
    """Weighted distributed QP with hard CBF collision rows."""
    st, sim = _load('distributed_qp')
    _apply_common(st, instance, params)
    st.d_safe_enforced, back_off = enforced_distance(instance, params)
    out = sim.run(out_dir=None, make_plots=False)
    return _to_result('dqp', out, {'back_off': back_off})


def run_cbf_omni(instance: BenchmarkInstance, params: dict, out_dir: Path) -> RunResult:
    """CBF-QP with heavily penalized soft safety rows; back-off replaces its 2*l buffer."""
    st, sim = _load('cbf_qp')
    _apply_common(st, instance, params)
    st.d_safe_enforced, back_off = enforced_distance(instance, params)
    out = sim.run(out_dir=None, make_plots=False)
    return _to_result('cbf_omni', out, {'back_off': back_off})


def _run_orca(method, folder, instance, params) -> RunResult:
    st, sim = _load(folder)
    _apply_common(st, instance, params)
    enforced, back_off = enforced_distance(instance, params)
    st.orca_radius = enforced / 2.0
    st.orca_max_speed = instance.config.v_max
    out = sim.run(out_dir=None, make_plots=False)
    return _to_result(method, out, {'back_off': back_off})


def run_orca_omni(instance: BenchmarkInstance, params: dict, out_dir: Path) -> RunResult:
    """Holonomic ORCA, combined radius = enforced distance."""
    return _run_orca('orca_omni', 'orca_radial_switching', instance, params)


def run_nh_orca_omni(instance: BenchmarkInstance, params: dict, out_dir: Path) -> RunResult:
    """The colleague's NH-ORCA folder.

    Its orca.py and node.py are identical to ``orca_radial_switching``'s; the
    only difference was the 2*epsilon radius inflation. Under the shared
    enforced distance it is therefore the same controller as ``orca_omni``.
    """
    return _run_orca('nh_orca_omni', 'nh_orca_radial_switching', instance, params)


def run_dhqp_omni(instance: BenchmarkInstance, params: dict, out_dir: Path) -> RunResult:
    """Distributed HQP (colleague's dhqp_radial_switching), omnidirectional model."""
    st, sim = _load('dhqp_radial_switching')
    _apply_common(st, instance, params)
    config = instance.config
    st.safety_distance, back_off = enforced_distance(instance, params)
    st.v_min = -config.v_max
    st.neighbor_limit = params.get('neighbor_limit', config.neighbor_limit)
    st.n_control = params.get('n_control', config.n_control)
    st.n_pred = params.get('n_pred', 0)
    st.type = 'omni'
    st.simulation = False
    st.save_data = False
    out = sim.run(out_dir=str(Path(out_dir).resolve() / 'scenario_out'), make_plots=False)
    return _to_result(
        'dhqp_omni',
        out,
        {
            'back_off': back_off,
            'n_control': st.n_control,
            'neighbor_limit': st.neighbor_limit,
        },
    )
