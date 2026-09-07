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
    }

    return RunResult(
        x_hist=out['x_hist'],
        u_hist=out['u_hist'],
        solve_times=solve_times,
        infeasible_count=0,
        meta=meta,
    )
