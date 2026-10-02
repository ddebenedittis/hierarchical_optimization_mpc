"""Collision-statistics figures for the long-simulation radial-switching sweep.

Reads the raw sweep folder written by radial_switching_unicycle/network_simulation.py
(<sweep>/<neighbor_limit>/<comm_range>/<run_id>/{node_*_data.csv, run_info.json}); no summary
CSV is needed. For every run the robot positions are rebuilt from each node's own state columns
and the same metrics as comparison/common/metrics.py are computed (collision events, max and
mean violation of the safety distance).

Writes next to this file:
  radial_switching_experiments.pdf           strip plot (one dot per run)
  radial_switching_experiments_boxplots.pdf  boxplots, grouped by communication range

usage: python3 make_figure.py [sweep_dir]    (default: newest folder in <repo>/out)
"""

import itertools
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[5]
OUT_STRIP = HERE / 'radial_switching_experiments.pdf'
OUT_BOX = HERE / 'radial_switching_experiments_boxplots.pdf'

SURFACE = '#fcfcfb'
INK, MUTED, AXIS, GRID = '#0b0b0b', '#898781', '#c9c7c1', '#e4e2dd'

# Categorical palette sampled from the original boxplot figure (1..7 neighbors).
NEIGHBOR_COLOR = {
    1: '#66AAD7',
    2: '#E89775',
    3: '#F4D079',
    4: '#B182BB',
    5: '#ADCD82',
    6: '#94D8F5',
    7: '#C77282',
}
NEIGHBORS = list(NEIGHBOR_COLOR)
RANGES = ['2.5', '5', '10']  # folder names
METRICS = [
    ('collision_events', 'Collision events'),
    ('max_violation', 'Max. violation [m]'),
    ('mean_violation', 'Mean. violation [m]'),
]

plt.rcParams.update(
    {
        'font.family': 'DejaVu Sans',
        'font.size': 8.5,
        'axes.edgecolor': AXIS,
        'axes.labelcolor': INK,
        'axes.titlesize': 9.5,
        'axes.titleweight': 'bold',
        'xtick.color': MUTED,
        'ytick.color': INK,
        'xtick.labelsize': 7.5,
        'ytick.labelsize': 8,
        'axes.linewidth': 0.8,
        'figure.facecolor': SURFACE,
        'axes.facecolor': SURFACE,
        'savefig.facecolor': SURFACE,
        'pdf.fonttype': 42,
    }
)


def run_metrics(run_dir):
    """Collision metrics of one run, or None if the run is incomplete."""
    info_path = run_dir / 'run_info.json'
    if not info_path.exists():
        return None
    cfg = json.loads(info_path.read_text())['config']
    n_robot = cfg['n_robot']
    pos = []
    for j in range(n_robot):
        f = run_dir / f'node_{j}_data.csv'
        if not f.exists():
            return None
        a = np.genfromtxt(f, delimiter=',', names=True)
        pos.append(np.c_[a[f'stateX_{j}'], a[f'stateY_{j}']])
    steps = min(len(p) for p in pos)
    pos = np.stack([p[:steps] for p in pos], axis=1)

    d_safe = cfg['safety_distance'] + cfg['back_off'] 
    events, viol = 0, []
    for i, j in itertools.combinations(range(n_robot), 2):
        dist = np.linalg.norm(pos[:, i] - pos[:, j], axis=1)
        hit = dist < d_safe
        events += int(hit[0]) + int(np.sum(hit[1:] & ~hit[:-1]))
        viol.append(d_safe - dist[hit])
    viol = np.concatenate(viol)
    return {
        'collision_events': events,
        'max_violation': float(viol.max()) if viol.size else 0.0,
        'mean_violation': float(viol.mean()) if viol.size else 0.0,
        'run_id': int(run_dir.name),
    }


def load(sweep):
    """{(neighbor_limit, comm_range): [metrics per run]}"""
    data = {}
    for nl, cr in itertools.product(NEIGHBORS, RANGES):
        runs = sorted((sweep / str(nl) / cr).glob('*/'), key=lambda p: int(p.name))
        ms = [m for m in map(run_metrics, runs) if m is not None]
        data[(nl, cr)] = ms
        print(f'  {nl} neighbors, range {cr}: {len(ms)} runs')
    return data


def metric_limits(data):
    lims = {}
    for key, _ in METRICS:
        hi = max((m[key] for ms in data.values() for m in ms), default=1.0)
        lims[key] = (0, hi * 1.05 if hi > 0 else 1.0)
    return lims


def strip_plot(data):
    lims = metric_limits(data)
    fig, axes = plt.subplots(
        len(METRICS), len(RANGES), figsize=(10.5, 2.5 * len(METRICS)),
        constrained_layout=True,
    )  # fmt: skip
    for mi, (key, label) in enumerate(METRICS):
        for ci, cr in enumerate(RANGES):
            ax = axes[mi, ci]
            for i, nl in enumerate(NEIGHBORS):
                for m in data[(nl, cr)]:
                    jitter = ((m['run_id'] * 7) % 9 - 4) * 0.045
                    ax.scatter(m[key], i + jitter, s=9, color=NEIGHBOR_COLOR[nl], alpha=0.6,
                               edgecolors='none', zorder=3)  # fmt: skip
            ax.set_ylim(len(NEIGHBORS) - 0.5, -0.5)
            ax.set_xlim(*lims[key])
            ax.set_yticks(range(len(NEIGHBORS)))
            ax.set_yticklabels(
                [f'{n} neighbor' + ('' if n == 1 else 's') if ci == 0 else '' for n in NEIGHBORS]
            )
            ax.tick_params(length=0)
            ax.grid(axis='x', color=GRID, lw=0.6, zorder=0)
            for s in ('top', 'right', 'left'):
                ax.spines[s].set_visible(False)
            if mi == 0:
                ax.set_title(f'Comm. range {cr} m', loc='left')
            ax.set_xlabel(label)
    fig.savefig(OUT_STRIP)
    plt.close(fig)
    print(f'wrote {OUT_STRIP.name}')


def box_plot(data):
    fig, axes = plt.subplots(len(METRICS), 1, figsize=(6.5, 7.2), sharex=True,
                             constrained_layout=True)  # fmt: skip
    n = len(NEIGHBORS)
    w = 0.8 / n
    for ax, (key, label) in zip(axes, METRICS):
        for ci, cr in enumerate(RANGES):
            for i, nl in enumerate(NEIGHBORS):
                vals = [m[key] for m in data[(nl, cr)]]
                if not vals:
                    continue
                pos = ci + (i - (n - 1) / 2) * w
                ax.boxplot(
                    vals, positions=[pos], widths=w * 0.92, patch_artist=True, showfliers=False,
                    boxprops=dict(facecolor=NEIGHBOR_COLOR[nl], edgecolor=INK, lw=0.6),
                    medianprops=dict(color=INK, lw=0.9),
                    whiskerprops=dict(color=INK, lw=0.6), capprops=dict(color=INK, lw=0.6),
                )  # fmt: skip
        ax.set_ylabel(label)
        ax.set_ylim(bottom=0)
        ax.grid(axis='y', color=GRID, lw=0.6, ls='--', zorder=0)
        ax.set_axisbelow(True)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
    axes[-1].set_xticks(range(len(RANGES)))
    axes[-1].set_xticklabels(RANGES, fontsize=8.5, color=INK)
    axes[-1].set_xlim(-0.5, len(RANGES) - 0.5)
    axes[-1].set_xlabel('Communication range [m]')
    handles = [
        Patch(facecolor=NEIGHBOR_COLOR[nl], edgecolor=INK, lw=0.6,
              label=f'{nl} neighbor' + ('' if nl == 1 else 's'))
        for nl in NEIGHBORS
    ]  # fmt: skip
    fig.get_layout_engine().set(rect=(0, 0.07, 1, 0.93))
    fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False, fontsize=8)
    fig.savefig(OUT_BOX)
    plt.close(fig)
    print(f'wrote {OUT_BOX.name}')


def main():
    if len(sys.argv) > 1:
        sweep = Path(sys.argv[1])
    else:
        sweep = max((p for p in (REPO / 'out').iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime)
    print(f'sweep: {sweep}')
    data = load(sweep)
    strip_plot(data)
    box_plot(data)


if __name__ == '__main__':
    main()
