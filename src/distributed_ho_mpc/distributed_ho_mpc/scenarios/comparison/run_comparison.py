"""Campaign CLI for the radial-switching multi-robot comparison.

Runs one or more control methods over a set of seeded (and optionally the
deterministic symmetric) benchmark instances and saves each run under
``<out>/<method_tag>/<seed>/``. With ``--preset colleague_omni`` it runs the
omnidirectional three-scenario benchmark instead and saves each run under
``<out>/<scenario>/<method_tag>/<seed>/``.

Each method is registered in METHOD_REGISTRY as a module exposing::

    def run_instance(instance: BenchmarkInstance, params: dict, out_dir: Path) -> RunResult

or as ``"module:function"`` for a function with that signature.

With ``--workers N`` every job runs in its own freshly forked process (method
settings modules are global state, and the dHQP Function cache would otherwise
grow across runs), N at a time. ``--pin-cpus`` pins job slot k to the k-th
listed CPU, so each concurrent job owns one physical core. Jobs are interleaved
across methods and scenarios, so every method sees the same contention.

Adapters must NOT call save_run themselves. This module is the single place
a run gets persisted, so every run -- regardless of method -- ends up with
the same trajectory.npz / run_info.json layout (see common/run_io.py).

Examples:
    python3 run_comparison.py --preset colleague_omni --scenarios uniform,asymmetric \\
        --methods pf,dqp,cbf_omni,orca_omni,nh_orca_omni,dhqp_omni,dhqp_omni@nc4 \\
        --params '{"dhqp_omni@nc4": {"n_control": 4}}' --seeds 0:15 \\
        --workers 6 --pin-cpus 0,2,4,6,8,10
    python3 run_comparison.py --methods dhqp,dwqp@k100 --seeds 0:20 --symmetric
    python3 run_comparison.py --methods cbf,orca --seeds 0,1,2 \\
        --params '{"cbf": {"gamma": 0.5}, "dwqp@k100": {"kappa": 100}}'
"""

from __future__ import annotations

import os

# Single-threaded BLAS/OpenMP in every job, set before numpy is first imported:
# the problems are tiny, and a multithreaded BLAS would let one job spill onto
# the cores reserved for the others.
for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_var, '1')

import argparse  # noqa: E402
import importlib  # noqa: E402
import json  # noqa: E402
import multiprocessing  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

# --- sys.path bootstrap so this also runs as `python3 run_comparison.py` ---
_FILE = Path(__file__).resolve()
try:
    import distributed_ho_mpc  # noqa: F401
except ImportError:
    for _root in (
        _FILE.parents[3],
        _FILE.parents[4] / 'hierarchical_optimization_mpc',
        _FILE.parents[4] / 'hierarchical_qp',
    ):
        sys.path.insert(0, str(_root))

from distributed_ho_mpc.scenarios.comparison.common.benchmark import (  # noqa: E402
    OMNI_SCENARIOS,
    colleague_omni,
    generate_instance,
    generate_omni_instance,
    symmetric_instance,
)
from distributed_ho_mpc.scenarios.comparison.common.run_io import save_run  # noqa: E402

_OMNI = 'distributed_ho_mpc.scenarios.comparison.methods.omni_adapters'

METHOD_REGISTRY = {
    'dhqp': 'distributed_ho_mpc.scenarios.comparison.methods.dhqp_adapter',
    'dwqp': 'distributed_ho_mpc.scenarios.comparison.methods.dwqp_adapter',
    'cbf': 'distributed_ho_mpc.scenarios.comparison.methods.cbf_qp',
    'orca': 'distributed_ho_mpc.scenarios.comparison.methods.nh_orca',
    # Omnidirectional benchmark (--preset colleague_omni), colleague's simulators.
    'pf': f'{_OMNI}:run_pf',
    'dqp': f'{_OMNI}:run_dqp',
    'cbf_omni': f'{_OMNI}:run_cbf_omni',
    'orca_omni': f'{_OMNI}:run_orca_omni',
    'nh_orca_omni': f'{_OMNI}:run_nh_orca_omni',
    'dhqp_omni': f'{_OMNI}:run_dhqp_omni',
}

PRESETS = {'colleague_omni': colleague_omni}


def parse_seeds(spec: str) -> list[int]:
    """Parse a seed spec: "start:stop" (exclusive range) or a comma list of ints."""
    if not spec:
        return []
    if ':' in spec:
        start, stop = spec.split(':', 1)
        return list(range(int(start), int(stop)))
    return [int(s) for s in spec.split(',') if s]


def parse_methods(spec: str) -> list[tuple[str, str]]:
    """Parse a comma-separated methods spec into (base_name, tag) pairs.

    "dwqp" -> ("dwqp", "dwqp"); "dwqp@k100" -> ("dwqp", "dwqp@k100"), where
    the tag becomes the output subdirectory name and the --params lookup key.
    """
    methods = []
    for item in spec.split(','):
        item = item.strip()
        if not item:
            continue
        base = item.split('@', 1)[0]
        methods.append((base, item))
    return methods


def default_out_root() -> Path:
    """Default output root: <repo_root>/out/comparison_<timestamp>/.

    repo_root is $HOMPC_WS if set, else parents[5] of this file.
    """
    repo_root = Path(os.environ['HOMPC_WS']) if os.environ.get('HOMPC_WS') else _FILE.parents[5]
    stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    return repo_root / 'out' / f'comparison_{stamp}'


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the run_comparison CLI argument parser."""
    parser = argparse.ArgumentParser(
        description='Radial-switching multi-robot comparison campaign.'
    )
    parser.add_argument(
        '--methods', required=True, help='Comma list, e.g. "dhqp,dwqp@k100,cbf,orca".'
    )
    parser.add_argument(
        '--seeds', default='', help='"start:stop" range or comma list, e.g. "0:50".'
    )
    parser.add_argument(
        '--symmetric', action='store_true', help='Also run the deterministic symmetric instance.'
    )
    parser.add_argument(
        '--preset',
        default=None,
        choices=sorted(PRESETS),
        help='Benchmark preset. colleague_omni runs the omnidirectional benchmark.',
    )
    parser.add_argument(
        '--scenarios',
        default=','.join(OMNI_SCENARIOS),
        help='Comma list of scenarios for --preset colleague_omni.',
    )
    parser.add_argument('--out', default=None, help='Output root directory.')
    parser.add_argument('--params', default='{}', help='JSON dict of per-method(-tag) param dicts.')
    parser.add_argument(
        '--force', action='store_true', help='Re-run and overwrite runs that already exist.'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=0,
        help='Concurrent job processes (0: run in this process, one after the other).',
    )
    parser.add_argument(
        '--pin-cpus',
        default='',
        help='Comma list of CPU ids; job slot k is pinned to the k-th (needs --workers).',
    )
    return parser


def resolve_method(base_method: str):
    """Return the run_instance callable registered for `base_method`."""
    target = METHOD_REGISTRY[base_method]
    module_path, _, attr = target.partition(':')
    module = importlib.import_module(module_path)
    return getattr(module, attr or 'run_instance')


def build_jobs(args) -> list[dict]:
    """Expand the CLI into an ordered job list.

    Order is seed-major, then scenario, then method, so consecutive jobs (and
    hence concurrently running ones) mix all methods.
    """
    seeds = parse_seeds(args.seeds)
    methods = parse_methods(args.methods)
    all_params = json.loads(args.params)

    unknown = [base for base, _ in methods if base not in METHOD_REGISTRY]
    for base in unknown:
        print(f'[run_comparison] Unknown method "{base}", skipping.')
    methods = [(base, tag) for base, tag in methods if base not in unknown]

    jobs = []
    if args.preset:
        base_config = PRESETS[args.preset]()
        scenarios = [s for s in args.scenarios.split(',') if s]
        for seed in seeds:
            for scenario in scenarios:
                instance = generate_omni_instance(seed, scenario, base_config)
                for base, tag in methods:
                    jobs.append(
                        {
                            'base': base,
                            'tag': tag,
                            'label': f'{scenario}/{tag}/{seed}',
                            'instance': instance,
                            'params': all_params.get(tag, all_params.get(base, {})),
                            'rel_dir': Path(scenario) / tag / str(seed),
                        }
                    )
        return jobs

    instances = [('-1', symmetric_instance())] if args.symmetric else []
    instances += [(str(seed), generate_instance(seed)) for seed in seeds]
    for base, tag in methods:
        for seed_label, instance in instances:
            jobs.append(
                {
                    'base': base,
                    'tag': tag,
                    'label': f'{tag}/{seed_label}',
                    'instance': instance,
                    'params': all_params.get(tag, all_params.get(base, {})),
                    'rel_dir': Path(tag) / seed_label,
                }
            )
    return jobs


def run_job(job: dict, out_dir: Path) -> bool:
    """Run one job in the current process and persist it. Returns success."""
    try:
        run_instance = resolve_method(job['base'])
        result = run_instance(job['instance'], job['params'], out_dir)
        save_run(out_dir, job['tag'], job['instance'], result, job['params'])
        n_steps = result.u_hist.shape[0]
        print(
            f'[run_comparison] {job["label"]}: '
            f'{n_steps} steps, infeasible={result.infeasible_count}',
            flush=True,
        )
        return True
    except Exception:
        print(f'[run_comparison] {job["label"]}: FAILED', flush=True)
        traceback.print_exc()
        return False


def _job_process(job: dict, out_dir: Path, cpu: int | None) -> None:
    """Body of a forked job process: pin, redirect output to the run's log, run."""
    if cpu is not None:
        os.sched_setaffinity(0, {cpu})
    out_dir.mkdir(parents=True, exist_ok=True)
    log = open(out_dir / 'log.txt', 'w', buffering=1)  # noqa: SIM115 -- lives as long as the process
    sys.stdout = sys.stderr = log
    print(f'[run_comparison] {job["label"]} on cpu {cpu}, pid {os.getpid()}', flush=True)
    ok = run_job(job, out_dir)
    log.flush()
    os._exit(0 if ok else 1)


def run_parallel(jobs: list[tuple[dict, Path]], workers: int, cpus: list[int]) -> None:
    """Run jobs in fresh forked processes, `workers` at a time, slot k on cpus[k]."""
    ctx = multiprocessing.get_context('fork')
    pending = list(jobs)
    running: dict[int, tuple] = {}  # slot -> (process, job, t0)
    n_done = n_failed = 0
    while pending or running:
        for slot in range(workers):
            if slot not in running and pending:
                job, out_dir = pending.pop(0)
                cpu = cpus[slot] if cpus else None
                proc = ctx.Process(target=_job_process, args=(job, out_dir, cpu))
                proc.start()
                running[slot] = (proc, job, time.perf_counter())
        time.sleep(0.2)
        for slot, (proc, job, t0) in list(running.items()):
            if proc.exitcode is None:
                continue
            proc.join()
            del running[slot]
            n_done += 1
            status = 'ok' if proc.exitcode == 0 else f'FAILED (exit {proc.exitcode})'
            n_failed += proc.exitcode != 0
            print(
                f'[run_comparison] [{n_done}/{len(jobs)}] {job["label"]}: {status} '
                f'in {time.perf_counter() - t0:.1f} s',
                flush=True,
            )
    print(f'[run_comparison] done: {n_done - n_failed} ok, {n_failed} failed', flush=True)


def main() -> None:
    """Entry point: parse args, build the instance/method matrix, and run it."""
    args = build_arg_parser().parse_args()
    out_root = Path(args.out) if args.out else default_out_root()
    cpus = [int(c) for c in args.pin_cpus.split(',') if c]
    if cpus and len(cpus) < args.workers:
        raise SystemExit(f'--pin-cpus lists {len(cpus)} CPUs for {args.workers} workers')

    todo = []
    for job in build_jobs(args):
        out_dir = out_root / job['rel_dir']
        if (out_dir / 'run_info.json').exists() and not args.force:
            print(f'[run_comparison] {job["label"]}: already exists, skipping.')
            continue
        todo.append((job, out_dir))
    print(f'[run_comparison] {len(todo)} jobs to run under {out_root}', flush=True)

    if args.workers > 0:
        run_parallel(todo, args.workers, cpus)
    else:
        for job, out_dir in todo:
            run_job(job, out_dir)


if __name__ == '__main__':
    main()
