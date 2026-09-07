"""Campaign CLI for the radial-switching multi-robot comparison.

Runs one or more control methods over a set of seeded (and optionally the
deterministic symmetric) benchmark instances and saves each run under
``<out>/<method_tag>/<seed>/``.

Each method is registered in METHOD_REGISTRY as a module exposing::

    def run_instance(instance: BenchmarkInstance, params: dict, out_dir: Path) -> RunResult

Adapters must NOT call save_run themselves. This module is the single place
a run gets persisted, so every run -- regardless of method -- ends up with
the same trajectory.npz / run_info.json layout (see common/run_io.py).

Examples:
    python3 run_comparison.py --methods dhqp,dwqp@k100 --seeds 0:20 --symmetric
    python3 run_comparison.py --methods cbf,orca --seeds 0,1,2 \\
        --params '{"cbf": {"gamma": 0.5}, "dwqp@k100": {"kappa": 100}}'
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

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
    generate_instance,
    symmetric_instance,
)
from distributed_ho_mpc.scenarios.comparison.common.run_io import save_run  # noqa: E402

METHOD_REGISTRY = {
    'dhqp': 'distributed_ho_mpc.scenarios.comparison.methods.dhqp_adapter',
    'dwqp': 'distributed_ho_mpc.scenarios.comparison.methods.dwqp_adapter',
    'cbf': 'distributed_ho_mpc.scenarios.comparison.methods.cbf_qp',
    'orca': 'distributed_ho_mpc.scenarios.comparison.methods.nh_orca',
}


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
    parser.add_argument('--out', default=None, help='Output root directory.')
    parser.add_argument('--params', default='{}', help='JSON dict of per-method(-tag) param dicts.')
    parser.add_argument(
        '--force', action='store_true', help='Re-run and overwrite runs that already exist.'
    )
    return parser


def main() -> None:
    """Entry point: parse args, build the instance/method matrix, and run it."""
    args = build_arg_parser().parse_args()

    seeds = parse_seeds(args.seeds)
    methods = parse_methods(args.methods)
    all_params = json.loads(args.params)
    out_root = Path(args.out) if args.out else default_out_root()

    instances = [('sym', symmetric_instance())] if args.symmetric else []
    instances += [(str(seed), generate_instance(seed)) for seed in seeds]

    for base_method, tag in methods:
        module_path = METHOD_REGISTRY.get(base_method)
        if module_path is None:
            print(f'[run_comparison] Unknown method "{base_method}", skipping.')
            continue
        try:
            module = importlib.import_module(module_path)
        except ImportError as exc:
            print(
                f'[run_comparison] Could not import method "{base_method}" ({module_path}): {exc}'
            )
            continue

        params = all_params.get(tag, all_params.get(base_method, {}))

        for seed_label, instance in instances:
            out_dir = out_root / tag / seed_label
            if (out_dir / 'run_info.json').exists() and not args.force:
                print(f'[run_comparison] {tag}/{seed_label}: already exists, skipping.')
                continue

            try:
                result = module.run_instance(instance, params, out_dir)
                save_run(out_dir, tag, instance, result, params)
                n_steps = result.u_hist.shape[0]
                print(
                    f'[run_comparison] {tag}/{seed_label}: '
                    f'{n_steps} steps, infeasible={result.infeasible_count}'
                )
            except Exception:
                print(f'[run_comparison] {tag}/{seed_label}: FAILED')
                traceback.print_exc()
                continue


if __name__ == '__main__':
    main()
