"""Aggregate and analyze a multi-robot control-method comparison campaign.

Walks a campaign directory laid out as ``<method_tag>/<seed>/{trajectory.npz, run_info.json}``,
computes KPIs for every run, writes summary CSVs and paired statistics against a baseline
method, and renders comparison figures and a markdown report.

Usage::

    python3 analyze_comparison.py <campaign_root> [--out <dir>] [--baseline dhqp]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Bootstrap sys.path so `distributed_ho_mpc` is importable when this file is run directly.
# Package root (.../src/distributed_ho_mpc) is 4 directory levels up from this file.
_PKG_ROOT = Path(__file__).resolve().parents[3]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from distributed_ho_mpc.scenarios.comparison.common.metrics import compute_kpis
from distributed_ho_mpc.scenarios.comparison.common.run_io import load_run

NUMERIC_KPI_KEYS = (
    'collision_events',
    'min_distance',
    'max_violation',
    'mean_violation',
    'n_reached',
    'makespan',
    'time_to_goal_mean',
    'path_length_ratio',
    'control_effort',
    'smoothness',
    'solve_time_mean',
    'solve_time_max',
    'infeasible_count',
    'sim_steps',
)

BOXPLOT_KEYS = (
    'collision_events',
    'max_violation',
    'min_distance',
    'makespan',
    'path_length_ratio',
    'control_effort',
    'smoothness',
    'solve_time_mean',
)

PAIRED_KEYS = (
    'collision_events',
    'max_violation',
    'makespan',
    'path_length_ratio',
    'control_effort',
    'smoothness',
    'solve_time_mean',
)

SYMMETRIC_SEED = -1


def _is_nan(value: Any) -> bool:
    """Return True if value is a float NaN (never raises on non-numeric input)."""
    return isinstance(value, float) and np.isnan(value)


def _color(index: int) -> Any:
    return plt.get_cmap('tab10')(index % 10)


def discover_runs(campaign_root: Path) -> dict[str, dict[int, Path]]:
    """Find <method_tag>/<seed>/ run directories under campaign_root."""
    layout: dict[str, dict[int, Path]] = {}
    if not campaign_root.is_dir():
        return layout
    for method_dir in sorted(p for p in campaign_root.iterdir() if p.is_dir()):
        if method_dir.name == 'analysis':
            continue
        seed_map: dict[int, Path] = {}
        for seed_dir in sorted(p for p in method_dir.iterdir() if p.is_dir()):
            try:
                seed = int(seed_dir.name)
            except ValueError:
                continue
            if (seed_dir / 'trajectory.npz').exists() and (seed_dir / 'run_info.json').exists():
                seed_map[seed] = seed_dir
        if seed_map:
            layout[method_dir.name] = seed_map
    return layout


def load_all(campaign_root: Path) -> tuple[dict[str, dict[int, dict]], list[dict], list[str]]:
    """Load every run, compute KPIs, and tag rows with method_tag.

    Returns (runs, rows, warnings) where runs[method_tag][seed] holds the raw run_info,
    arrays, and kpi dict; rows is the flat list of per-(method_tag, seed) KPI dicts.
    """
    layout = discover_runs(campaign_root)
    runs: dict[str, dict[int, dict]] = {}
    rows: list[dict] = []
    warnings_list: list[str] = []
    for method_tag, seed_map in layout.items():
        runs[method_tag] = {}
        for seed, run_dir in seed_map.items():
            try:
                run_info, arrays = load_run(run_dir)
                kpi = compute_kpis(run_info, arrays)
            except Exception as exc:
                msg = f'WARN: failed to load run {run_dir}: {exc}'
                print(msg)
                warnings_list.append(msg)
                continue
            runs[method_tag][seed] = {'run_info': run_info, 'arrays': arrays, 'kpi': kpi}
            row = dict(kpi)
            row['method_tag'] = method_tag
            row.setdefault('seed', seed)
            rows.append(row)
    return runs, rows, warnings_list


def write_summary_csv(rows: list[dict], out_path: Path) -> list[str]:
    """Write one row per (method_tag, seed) with all KPI columns."""
    if not rows:
        out_path.write_text('')
        return []
    field_set: set[str] = set()
    for row in rows:
        field_set.update(row.keys())
    field_set.discard('method_tag')
    field_set.discard('seed')
    fieldnames = ['method_tag', 'seed'] + sorted(field_set)
    with out_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for row in sorted(rows, key=lambda r: (str(r.get('method_tag', '')), r.get('seed', 0))):
            writer.writerow(row)
    return fieldnames


def compute_method_summary(rows: list[dict], method_tag: str) -> dict[str, Any]:
    """Median/IQR of numeric KPIs, success_rate, and total collisions for one method_tag.

    Seed SYMMETRIC_SEED (-1) is excluded from these aggregate statistics.
    """
    method_rows = [
        r for r in rows if r.get('method_tag') == method_tag and r.get('seed') != SYMMETRIC_SEED
    ]
    summary: dict[str, Any] = {'method_tag': method_tag, 'n_seeds': len(method_rows)}

    success_vals = [bool(r['success']) for r in method_rows if r.get('success') is not None]
    summary['success_rate'] = float(np.mean(success_vals)) if success_vals else float('nan')

    collision_vals = [
        r['collision_events']
        for r in method_rows
        if r.get('collision_events') is not None and not _is_nan(r['collision_events'])
    ]
    summary['total_collision_events'] = float(np.sum(collision_vals)) if collision_vals else 0.0

    for key in NUMERIC_KPI_KEYS:
        vals = [r[key] for r in method_rows if r.get(key) is not None and not _is_nan(r[key])]
        if vals:
            summary[f'median_{key}'] = float(np.median(vals))
            summary[f'q25_{key}'] = float(np.percentile(vals, 25))
            summary[f'q75_{key}'] = float(np.percentile(vals, 75))
        else:
            summary[f'median_{key}'] = float('nan')
            summary[f'q25_{key}'] = float('nan')
            summary[f'q75_{key}'] = float('nan')
    return summary


def write_summary_by_method_csv(
    rows: list[dict], method_tags: list[str], out_path: Path
) -> list[dict[str, Any]]:
    summaries = [compute_method_summary(rows, mt) for mt in method_tags]
    if not summaries:
        out_path.write_text('')
        return []
    fieldnames = ['method_tag', 'n_seeds', 'success_rate', 'total_collision_events']
    for key in NUMERIC_KPI_KEYS:
        fieldnames += [f'median_{key}', f'q25_{key}', f'q75_{key}']
    with out_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in summaries:
            writer.writerow(s)
    return summaries


def sanity_checks(runs: dict[str, dict[int, dict]]) -> list[str]:
    """Cross-method consistency checks. Prints WARN lines and returns them; never raises."""
    warnings_list: list[str] = []

    seeds: set[int] = set()
    for seed_map in runs.values():
        seeds.update(seed_map.keys())

    for seed in sorted(seeds):
        entries = [(mt, runs[mt][seed]) for mt in runs if seed in runs[mt]]
        if len(entries) < 2:
            continue
        ref_tag, ref_data = entries[0]
        ref_s_init = ref_data['arrays'].get('s_init')
        ref_goals = ref_data['arrays'].get('goals')
        for tag, data in entries[1:]:
            s_init = data['arrays'].get('s_init')
            goals = data['arrays'].get('goals')
            if ref_s_init is not None and s_init is not None:
                if ref_s_init.shape != s_init.shape or not np.allclose(ref_s_init, s_init):
                    msg = f'WARN seed {seed}: s_init mismatch between {ref_tag} and {tag}'
                    print(msg)
                    warnings_list.append(msg)
            if ref_goals is not None and goals is not None:
                if ref_goals.shape != goals.shape or not np.allclose(ref_goals, goals):
                    msg = f'WARN seed {seed}: goals mismatch between {ref_tag} and {tag}'
                    print(msg)
                    warnings_list.append(msg)

    tol = 1e-6
    for method_tag, seed_map in runs.items():
        for seed, data in seed_map.items():
            cfg = data['run_info'].get('config', {}) or {}
            u_hist = data['arrays'].get('u_hist')
            if u_hist is None or u_hist.size == 0:
                continue
            bounds = (
                ('v_min', 0, np.less, 'below'),
                ('v_max', 0, np.greater, 'above'),
                ('omega_min', 1, np.less, 'below'),
                ('omega_max', 1, np.greater, 'above'),
            )
            for cfg_key, col, cmp_fn, direction in bounds:
                limit = cfg.get(cfg_key)
                if limit is None or u_hist.shape[-1] <= col:
                    continue
                offset = -tol if direction == 'below' else tol
                if np.any(cmp_fn(u_hist[..., col], limit + offset)):
                    msg = (
                        f'WARN {method_tag}/seed {seed}: u_hist column {col} exceeds '
                        f'{cfg_key}={limit} ({direction} tolerance {tol})'
                    )
                    print(msg)
                    warnings_list.append(msg)
    return warnings_list


def compute_paired_stats(
    rows: list[dict], method_tags: list[str], baseline: str
) -> list[dict[str, Any]]:
    """Wilcoxon signed-rank test of each method against baseline, per KPI, on matched seeds.

    Seed SYMMETRIC_SEED is excluded (it is a single deterministic instance, not a sample).
    """
    by_method_seed: dict[str, dict[int, dict]] = {}
    for r in rows:
        by_method_seed.setdefault(r['method_tag'], {})[r.get('seed')] = r

    baseline_data = by_method_seed.get(baseline, {})
    results: list[dict[str, Any]] = []
    if not baseline_data:
        print(f'WARN: baseline method_tag "{baseline}" not found; skipping paired statistics')
        return results

    for mt in method_tags:
        if mt == baseline:
            continue
        method_data = by_method_seed.get(mt, {})
        common_seeds = sorted(
            s for s in (set(baseline_data) & set(method_data)) if s != SYMMETRIC_SEED
        )

        for key in PAIRED_KEYS:
            base_vals: list[float] = []
            meth_vals: list[float] = []
            for s in common_seeds:
                bv = baseline_data[s].get(key)
                mv = method_data[s].get(key)
                if bv is None or mv is None or _is_nan(bv) or _is_nan(mv):
                    continue
                base_vals.append(bv)
                meth_vals.append(mv)
            n = len(base_vals)
            entry: dict[str, Any] = {
                'method_tag': mt,
                'kpi': key,
                'n': n,
                'median_baseline': float(np.median(base_vals)) if n else float('nan'),
                'median_method': float(np.median(meth_vals)) if n else float('nan'),
                'p_value': '',
                'note': '',
            }
            if n < 5:
                entry['note'] = 'skipped: n<5'
            elif np.allclose(np.array(meth_vals) - np.array(base_vals), 0.0):
                entry['note'] = 'skipped: all differences zero'
            else:
                try:
                    _, p = stats.wilcoxon(
                        base_vals, meth_vals, zero_method='wilcox', alternative='two-sided'
                    )
                    entry['p_value'] = float(p)
                except ValueError as exc:
                    entry['note'] = f'skipped: {exc}'
            results.append(entry)

        succ_base = [
            baseline_data[s]['success']
            for s in common_seeds
            if baseline_data[s].get('success') is not None
        ]
        succ_meth = [
            method_data[s]['success']
            for s in common_seeds
            if method_data[s].get('success') is not None
        ]
        results.append(
            {
                'method_tag': mt,
                'kpi': 'success',
                'n': len(common_seeds),
                'median_baseline': float(np.mean(succ_base)) if succ_base else float('nan'),
                'median_method': float(np.mean(succ_meth)) if succ_meth else float('nan'),
                'p_value': '',
                'note': 'rate comparison only',
            }
        )
    return results


def write_stats_csv(results: list[dict[str, Any]], out_path: Path) -> None:
    fieldnames = ['method_tag', 'kpi', 'n', 'median_baseline', 'median_method', 'p_value', 'note']
    with out_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)


def plot_boxplots(rows: list[dict], method_tags: list[str], out_path: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for ax, key in zip(axes.flat, BOXPLOT_KEYS):
        data = []
        labels = []
        for mt in method_tags:
            vals = [
                r[key]
                for r in rows
                if r.get('method_tag') == mt and r.get(key) is not None and not _is_nan(r[key])
            ]
            data.append(vals)
            labels.append(mt)
        try:
            bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)
            for i, box in enumerate(bp['boxes']):
                box.set_facecolor(_color(i))
                box.set_alpha(0.6)
        except Exception as exc:
            print(f'WARN: could not draw boxplot for {key}: {exc}')
        ax.set_title(key)
        ax.tick_params(axis='x', rotation=45)
        if key == 'solve_time_mean':
            ax.set_yscale('log')
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


def plot_pareto(rows: list[dict], method_tags: list[str], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, mt in enumerate(method_tags):
        pts = [
            (r['max_violation'], r['makespan'])
            for r in rows
            if r.get('method_tag') == mt
            and r.get('max_violation') is not None
            and r.get('makespan') is not None
            and not _is_nan(r['max_violation'])
            and not _is_nan(r['makespan'])
        ]
        if not pts:
            continue
        xs, ys = zip(*pts)
        color = _color(i)
        ax.scatter(xs, ys, color=color, alpha=0.25, s=25)
        ax.scatter(
            np.median(xs),
            np.median(ys),
            color=color,
            s=180,
            marker='D',
            edgecolor='black',
            linewidth=1,
            label=mt,
            zorder=5,
        )
    ax.set_xlabel('max_violation')
    ax.set_ylabel('makespan')
    ax.set_title('Safety vs. performance (points = seeds, diamonds = per-method median)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


def _baseline_median_makespan_seed(runs: dict[str, dict[int, dict]], baseline: str) -> int | None:
    baseline_runs = runs.get(baseline, {})
    candidates = [
        (seed, data['kpi'].get('makespan'))
        for seed, data in baseline_runs.items()
        if seed != SYMMETRIC_SEED
        and data['kpi'].get('makespan') is not None
        and not _is_nan(data['kpi']['makespan'])
    ]
    if not candidates:
        return None
    median_val = float(np.median([m for _, m in candidates]))
    return min(candidates, key=lambda t: abs(t[1] - median_val))[0]


def _min_pairwise_distance(pos: np.ndarray) -> float:
    """pos: (N, 2) positions. Returns min pairwise distance (inf if N < 2)."""
    n = pos.shape[0]
    if n < 2:
        return float('inf')
    diffs = pos[:, None, :] - pos[None, :, :]
    dist = np.linalg.norm(diffs, axis=-1)
    np.fill_diagonal(dist, np.inf)
    return float(dist.min())


def plot_min_distance_time(
    runs: dict[str, dict[int, dict]],
    method_tags: list[str],
    baseline: str,
    out_path: Path,
) -> int | None:
    seed = _baseline_median_makespan_seed(runs, baseline)
    if seed is None:
        print('WARN: could not determine baseline median-makespan seed; skipping min_distance_time')
        return None

    fig, ax = plt.subplots(figsize=(9, 5))
    safety_distance = None
    for i, mt in enumerate(method_tags):
        data = runs.get(mt, {}).get(seed)
        if data is None:
            continue
        x_hist = data['arrays'].get('x_hist')
        if x_hist is None or x_hist.shape[1] < 2:
            continue
        cfg = data['run_info'].get('config', {}) or {}
        dt = cfg.get('dt', data['run_info'].get('dt', 1.0))
        if safety_distance is None:
            safety_distance = cfg.get('safety_distance')
        t_steps = x_hist.shape[0]
        min_d = np.array([_min_pairwise_distance(x_hist[t, :, :2]) for t in range(t_steps)])
        time = np.arange(t_steps) * dt
        ax.plot(time, min_d, color=_color(i), label=mt)

    if safety_distance is not None:
        ax.axhline(safety_distance, color='red', linestyle='--', label='safety_distance')
    ax.set_xlabel('time [s]')
    ax.set_ylabel('min inter-robot distance')
    ax.set_title(f'Minimum inter-robot distance (seed {seed}, baseline median-makespan)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    return seed


def plot_trajectories(
    runs: dict[str, dict[int, dict]], method_tags: list[str], seed: int, out_path: Path
) -> None:
    present = [mt for mt in method_tags if seed in runs.get(mt, {})]
    if not present:
        print(f'WARN: no runs available for seed {seed}; skipping trajectories figure')
        return
    fig, axes = plt.subplots(1, len(present), figsize=(4.5 * len(present), 4.5), squeeze=False)
    axes = axes[0]
    for ax, mt in zip(axes, present):
        data = runs[mt][seed]
        arrays = data['arrays']
        x_hist = arrays.get('x_hist')
        goals = arrays.get('goals')
        s_init = arrays.get('s_init')
        if x_hist is None:
            ax.set_title(f'{mt} (no data)')
            continue
        n_robots = x_hist.shape[1]
        for r in range(n_robots):
            color = _color(r)
            ax.plot(x_hist[:, r, 0], x_hist[:, r, 1], color=color, alpha=0.6, linewidth=1)
            if s_init is not None:
                ax.scatter(s_init[r, 0], s_init[r, 1], color=color, marker='o', s=40, zorder=3)
            if goals is not None:
                ax.scatter(goals[r, 0], goals[r, 1], color=color, marker='x', s=50, zorder=3)
        ax.set_title(mt)
        ax.set_aspect('equal', adjustable='datalim')
    fig.suptitle(f'Trajectories (seed {seed})')
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


def markdown_table(fieldnames: list[str], rows: list[dict[str, Any]]) -> str:
    if not rows:
        return '(no data)\n'

    def fmt(v: Any) -> str:
        if isinstance(v, float):
            return f'{v:.4g}'
        return str(v)

    lines = ['| ' + ' | '.join(fieldnames) + ' |']
    lines.append('| ' + ' | '.join('---' for _ in fieldnames) + ' |')
    for row in rows:
        lines.append('| ' + ' | '.join(fmt(row.get(fn, '')) for fn in fieldnames) + ' |')
    return '\n'.join(lines) + '\n'


def write_report(
    campaign_root: Path,
    out_dir: Path,
    runs: dict[str, dict[int, dict]],
    method_tags: list[str],
    summaries: list[dict[str, Any]],
    stats_results: list[dict[str, Any]],
    sanity_warnings: list[str],
    load_warnings: list[str],
    baseline: str,
) -> None:
    lines: list[str] = []
    lines.append('# Comparison campaign report')
    lines.append('')
    lines.append(f'Campaign root: `{campaign_root}`')
    lines.append(f'Baseline method_tag: `{baseline}`')
    lines.append('')
    lines.append('## Run counts')
    lines.append('')
    for mt in method_tags:
        n = len(runs.get(mt, {}))
        n_sym = 1 if SYMMETRIC_SEED in runs.get(mt, {}) else 0
        lines.append(f'- `{mt}`: {n} runs ({n_sym} symmetric stress instance included)')
    lines.append('')

    lines.append('## Per-method summary (medians and IQR, excluding the symmetric instance)')
    lines.append('')
    if summaries:
        key_cols = ['method_tag', 'n_seeds', 'success_rate', 'total_collision_events']
        for key in ('makespan', 'max_violation', 'collision_events', 'solve_time_mean'):
            key_cols.append(f'median_{key}')
        key_cols = [c for c in key_cols if any(c in s for s in summaries)]
        lines.append(markdown_table(key_cols, summaries))
    else:
        lines.append('(no runs found)')
    lines.append('')

    lines.append(f'## Paired statistics vs. baseline `{baseline}`')
    lines.append('')
    if stats_results:
        lines.append(
            markdown_table(
                ['method_tag', 'kpi', 'n', 'median_baseline', 'median_method', 'p_value', 'note'],
                stats_results,
            )
        )
    else:
        lines.append('(no paired statistics available)')
    lines.append('')

    lines.append('## Sanity warnings')
    lines.append('')
    all_warnings = load_warnings + sanity_warnings
    if all_warnings:
        for w in all_warnings:
            lines.append(f'- {w}')
    else:
        lines.append('None.')
    lines.append('')

    lines.append('## Symmetric stress instance (seed -1)')
    lines.append('')
    sym_rows = []
    for mt in method_tags:
        data = runs.get(mt, {}).get(SYMMETRIC_SEED)
        if data is None:
            continue
        kpi = data['kpi']
        sym_rows.append(
            {
                'method_tag': mt,
                'success': kpi.get('success'),
                'makespan': kpi.get('makespan'),
                'collision_events': kpi.get('collision_events'),
                'min_distance': kpi.get('min_distance'),
            }
        )
    if sym_rows:
        lines.append(
            markdown_table(
                ['method_tag', 'success', 'makespan', 'collision_events', 'min_distance'],
                sym_rows,
            )
        )
    else:
        lines.append('No seed -1 (symmetric) runs found for any method.')
    lines.append('')

    lines.append('## Figures')
    lines.append('')
    lines.append('See `boxplots.pdf`, `pareto.pdf`, `min_distance_time.pdf`, and')
    lines.append('`trajectories_seed<S>.pdf` in this directory.')
    lines.append('')

    (out_dir / 'report.md').write_text('\n'.join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description='Analyze a comparison campaign.')
    parser.add_argument('campaign_root', type=Path, help='Root directory of the campaign.')
    parser.add_argument(
        '--out',
        type=Path,
        default=None,
        help='Output directory (default: <campaign_root>/analysis).',
    )
    parser.add_argument('--baseline', type=str, default='dhqp', help='Baseline method_tag.')
    args = parser.parse_args()

    campaign_root: Path = args.campaign_root.resolve()
    out_dir: Path = (args.out or campaign_root / 'analysis').resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'Loading campaign from {campaign_root} ...')
    runs, rows, load_warnings = load_all(campaign_root)
    method_tags = sorted(runs.keys())
    print(f'Found method_tags: {method_tags}')
    print(f'Loaded {len(rows)} runs total.')

    write_summary_csv(rows, out_dir / 'summary.csv')
    summaries = write_summary_by_method_csv(rows, method_tags, out_dir / 'summary_by_method.csv')

    sanity_warnings = sanity_checks(runs)

    stats_results = compute_paired_stats(rows, method_tags, args.baseline)
    write_stats_csv(stats_results, out_dir / 'stats_vs_baseline.csv')

    if rows:
        try:
            plot_boxplots(rows, method_tags, out_dir / 'boxplots.pdf')
        except Exception as exc:
            print(f'WARN: failed to render boxplots.pdf: {exc}')
        try:
            plot_pareto(rows, method_tags, out_dir / 'pareto.pdf')
        except Exception as exc:
            print(f'WARN: failed to render pareto.pdf: {exc}')

        seed = None
        try:
            seed = plot_min_distance_time(
                runs, method_tags, args.baseline, out_dir / 'min_distance_time.pdf'
            )
        except Exception as exc:
            print(f'WARN: failed to render min_distance_time.pdf: {exc}')

        if seed is None:
            seed = _baseline_median_makespan_seed(runs, args.baseline)
        if seed is not None:
            try:
                plot_trajectories(runs, method_tags, seed, out_dir / f'trajectories_seed{seed}.pdf')
            except Exception as exc:
                print(f'WARN: failed to render trajectories_seed{seed}.pdf: {exc}')
        else:
            print('WARN: no valid baseline seed found; skipping trajectories figure')
    else:
        print('WARN: no runs loaded; skipping all figures')

    write_report(
        campaign_root,
        out_dir,
        runs,
        method_tags,
        summaries,
        stats_results,
        sanity_warnings,
        load_warnings,
        args.baseline,
    )
    print(f'Analysis written to {out_dir}')


if __name__ == '__main__':
    main()
