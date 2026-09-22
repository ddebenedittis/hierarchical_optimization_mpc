"""Adapter that runs the radial-switching dHQP scenario on injected benchmark instances.

Wraps ``radial_switching_unicycle.network_simulation.main`` so it can be driven by the
comparison campaign: settings are synced to the benchmark config, the scenario's own
random instance generation is bypassed in favor of the injected seed's ``s_init`` /
``goals``, and its return dict is repackaged as a ``RunResult``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from distributed_ho_mpc.scenarios.comparison.common.benchmark import BenchmarkInstance
from distributed_ho_mpc.scenarios.comparison.common.run_io import RunResult


def run_instance(instance: BenchmarkInstance, params: dict, out_dir: Path) -> RunResult:
    """Run the dHQP radial-switching scenario on one benchmark instance.

    Args:
        instance: Seeded benchmark instance (s_init, goals, config).
        params: Method params; may override 'comm_range' / 'limit_connection'.
        out_dir: Per-run output directory; the scenario writes its own artifacts
            under out_dir / 'scenario_out'.

    Returns:
        The RunResult built from the scenario's trajectory/timing output.
    """
    import distributed_ho_mpc.scenarios.radial_switching_unicycle.network_simulation as sim
    import distributed_ho_mpc.scenarios.radial_switching_unicycle.settings as st

    config = instance.config

    st.dt = config.dt
    st.v_max = config.v_max
    st.v_min = config.v_min
    st.omega_max = config.omega_max
    st.omega_min = config.omega_min
    st.n_nodes = config.n_robots
    st.n_steps = config.max_steps
    st.save_data = False

    # Center-to-center distance the collision task enforces. Both baselines
    # inflate their own barrier radius above config.safety_distance for concrete
    # geometric reasons -- CBF by 2*l for the feedback-linearization offset
    # point, ORCA by 2*epsilon for holonomic tracking error -- so dHQP was the
    # only method scored against a 2.0 bound while enforcing exactly 2.0, with
    # no room for its own discretization and slack error. `margin` gives it the
    # same explicit latitude, and defaults to 0.0 so nothing changes unless a
    # campaign sweeps it.
    margin = params.get('margin', 0.0)
    st.safety_distance = config.safety_distance + margin

    # MPC horizon. The scenario ships n_control = 1, n_pred = 0, which runs this
    # MPC as a one-step reactive controller -- the baselines get their gains
    # swept, so the horizon should be sweepable too. n_xi is a DERIVED primal
    # dimension that settings.py computes at import time, so it has to be
    # recomputed here or the variable dimensions silently disagree.
    st.n_control = params.get('n_control', st.n_control)
    st.n_pred = params.get('n_pred', st.n_pred)
    st.n_xi = st.n_control * (5 if st.type == 'uni' else 4)

    out = sim.main(
        config.n_robots,
        params.get('comm_range', config.comm_range),
        params.get('limit_connection', config.limit_connection),
        'comparison',
        int(instance.seed),
        s_init=[np.array(r) for r in instance.s_init],
        goals=[np.array(g) for g in instance.goals],
        max_steps=config.max_steps,
        out_dir_override=str(out_dir / 'scenario_out'),
        media=False,
        goal_tol=config.goal_tol,
    )

    steps = out['steps']
    solve_times = out.get('solve_times')
    if solve_times is None:
        solve_times = np.zeros((steps, config.n_robots))
    meta = {
        'solve_time_total': out['solve_time_total'],
        'solve_time_max': out['solve_time_max'],
        'n_solves': out['n_solves'],
        'creation_time_total': out.get('creation_time_total', 0.0),
        'infeasible_measured': False,
        'enforced_safety_distance': float(st.safety_distance),
    }

    return RunResult(
        x_hist=out['x_hist'],
        u_hist=out['u_hist'],
        solve_times=solve_times,
        infeasible_count=0,
        meta=meta,
    )
