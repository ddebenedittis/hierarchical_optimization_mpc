"""Run persistence: save/load simulation results and per-run metadata."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from distributed_ho_mpc.scenarios.comparison.common.benchmark import BenchmarkInstance


@dataclass
class RunResult:
    """Result of one simulation run, ready to be persisted by save_run.

    Attributes:
        x_hist: State history, shape (T+1, N, 3) for unicycles, (T+1, N, 2) for
            omnidirectional robots.
        u_hist: Input history, shape (T, N, 2).
        solve_times: Per-agent solve time in seconds, shape (T, N).
            All-zeros is allowed when only run-level aggregates are known
            (see ``meta``).
        infeasible_count: Total number of infeasible solves across the run.
        meta: Optional aggregates, e.g. solve_time_total, solve_time_max,
            n_solves, wall_time.
        arrays: Extra per-run arrays saved alongside the trajectory, e.g.
            qp_times (T, N) or link_events (T, 2).
    """

    x_hist: np.ndarray
    u_hist: np.ndarray
    solve_times: np.ndarray
    infeasible_count: int = 0
    meta: dict = field(default_factory=dict)
    arrays: dict = field(default_factory=dict)


def save_run(
    out_dir: Path,
    method: str,
    instance: BenchmarkInstance,
    result: RunResult,
    params: dict | None = None,
) -> None:
    """Persist one run's trajectory arrays and metadata under out_dir.

    Writes ``trajectory.npz`` (x_hist, u_hist, solve_times, s_init, goals)
    and ``run_info.json`` (method, seed, params, infeasible_count, meta,
    config, n_robots, dt).

    Args:
        out_dir: Per-run output directory (created if missing, parents ok).
        method: Method name/tag under which this run is recorded.
        instance: The benchmark instance that was run.
        result: The RunResult to persist.
        params: Params dict used for this run.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    params = params if params is not None else {}

    np.savez(
        out_dir / 'trajectory.npz',
        x_hist=result.x_hist,
        u_hist=result.u_hist,
        solve_times=result.solve_times,
        s_init=instance.s_init,
        goals=instance.goals,
        **result.arrays,
    )

    run_info = {
        'method': method,
        'seed': instance.seed,
        'params': params,
        'infeasible_count': result.infeasible_count,
        'meta': result.meta,
        'scenario': instance.config.scenario,
        'config': asdict(instance.config),
        'n_robots': instance.config.n_robots,
        'dt': instance.config.dt,
    }
    (out_dir / 'run_info.json').write_text(json.dumps(run_info, indent=2), encoding='utf-8')


def load_run(run_dir: Path) -> tuple[dict, dict]:
    """Load a run previously written by save_run.

    Args:
        run_dir: The per-run directory containing run_info.json and
            trajectory.npz.

    Returns:
        Tuple of (run_info dict, dict mapping array name -> ndarray from
        trajectory.npz).
    """
    run_dir = Path(run_dir)
    run_info = json.loads((run_dir / 'run_info.json').read_text(encoding='utf-8'))
    with np.load(run_dir / 'trajectory.npz') as npz:
        arrays = {key: npz[key] for key in npz.files}
    return run_info, arrays
