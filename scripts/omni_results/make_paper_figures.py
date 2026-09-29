"""Paper-style figures (PDF) for the omnidirectional radial-switching comparison.

Same inputs as make_results.py (per-run summaries and trajectory.npz of the two campaigns), drawn
with the paper's own style (init_matplotlib of hierarchical_optimization_mpc/utils/
disp_het_multi_rob.py: Times via usetex, MATLAB palette, dash-dot grid). Writes *.pdf to
out/omni_results/paper_figures/ by default.

usage: python3 make_paper_figures.py [--seed N] [--sources DIR [DIR ...]] [--out-dir DIR]
"""

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from make_results import CAP_S, D_SAFE, ENFORCED, OUT_DIR, ROOT, SCENARIOS, SOURCES
from matplotlib import colormaps
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch

OUT = OUT_DIR / 'paper_figures'
# Source tree first, so that the host Python needs neither ROS nor casadi.
sys.path.insert(0, str(ROOT / 'src' / 'hierarchical_optimization_mpc'))
from hierarchical_optimization_mpc.utils.disp_het_multi_rob import init_matplotlib  # noqa: E402

# (tag, label, color, linestyle). NH-ORCA is omitted: same code as ORCA, identical on every run.
METHODS = [
    ('dhqp_omni@nl5', 'dHQP (5 neighbors)', '#0072BD', '-'),
    ('dhqp_omni', 'dHQP (2 neighbors)', '#4DBEEE', '--'),
    ('pf', 'Potential field', '#D95319', '-.'),
    ('dqp', 'Distributed QP', '#EDB120', ':'),
    ('cbf_omni', 'CBF-QP', '#7E2F8E', '-'),
    ('orca_omni', 'ORCA', '#77AC30', '--'),
]
SCEN_LABEL = dict(SCENARIOS)
SCEN_KEYS = [k for k, _ in SCENARIOS]

COLLISION_MARGIN = 0.05  # paper: a collision is a violation of d_safe by at least 5 cm
D_FORM, FORM_TOL, GOAL_TOL = 2.0, 0.05, 0.01
# Formation pair (0, 1) of priority_conflict. dHQP ranks agent 0's goal above the formation and
# agent 1's formation above its goal; the weighted baselines' settings (pf, dqp, cbf_qp
# priority/weight_overrides) give agent 0 the formation-dominant weights and agent 1 the
# goal-dominant ones. Figures therefore show the agents by role: (goal-first, formation-first).
ROLES = {
    'dhqp_omni@nl5': (0, 1),
    'dhqp_omni': (0, 1),
    'orca_omni': (0, 1),
    'pf': (1, 0),
    'dqp': (1, 0),
    'cbf_omni': (1, 0),
}


# =================================== Data =================================== #


def load_runs(sources):
    """All runs, keyed by (scenario, tag), each a seed-sorted list of summary rows plus arrays."""
    runs = {}
    for src in sources:
        for scen in SCEN_KEYS:
            with open(src / 'analysis' / f'summary_{scen}.csv') as f:
                for r in csv.DictReader(f):
                    tag = r['method_tag']
                    if tag not in dict((m[0], m) for m in METHODS):
                        continue
                    run_dir = src / scen / tag / r['seed']
                    info = json.loads((run_dir / 'run_info.json').read_text())
                    arr = np.load(run_dir / 'trajectory.npz')
                    last = int(r['last_step'])

                    def num(k):
                        return float('nan') if r[k] in ('', 'nan') else float(r[k])

                    runs.setdefault((scen, tag), []).append(
                        {
                            'seed': int(r['seed']),
                            'converged': r['converged'] == 'True',
                            'safe_success': r['safe_success'] == 'True',
                            'deadlock': r['deadlock'] == 'True',
                            'norm_ttg': num('norm_ttg'),
                            'min_distance': num('min_distance'),
                            'ms': num('ms_per_agent_step'),
                            'qp_ms': num('qp_ms_per_agent_step'),
                            'last_step': last,
                            'dt': info['dt'],
                            'x': arr['x_hist'][: last + 1],
                            'goals': arr['goals'],
                        }
                    )
    for v in runs.values():
        v.sort(key=lambda r: r['seed'])
    return runs


def pair_distances(x):
    """(n_pairs, n_steps) distances of every robot pair."""
    return np.array(
        [
            np.linalg.norm(x[:, i] - x[:, j], axis=1)
            for i, j in itertools.combinations(range(x.shape[1]), 2)
        ]
    )


def collision_events(x, threshold):
    """Maximal runs of steps with a pair closer than threshold, counted once per pair."""
    inside = pair_distances(x) < threshold
    return int(np.sum(inside[:, 1:] & ~inside[:, :-1]) + np.sum(inside[:, 0]))


def max_violation(x):
    return float(max(0.0, D_SAFE - pair_distances(x).min()))


def representative_seeds(runs, override):
    """Per scenario, the seed at the median normalized time-to-goal of dHQP (5 neighbors)."""
    if override is not None:
        return {s: override for s in SCEN_KEYS}
    seeds = {}
    for s in SCEN_KEYS:
        rs = sorted(
            (r for r in runs[(s, METHODS[0][0])] if r['converged']), key=lambda r: r['norm_ttg']
        )
        seeds[s] = rs[len(rs) // 2]['seed']
    return seeds


def run_of(runs, scen, tag, seed):
    return next(r for r in runs[(scen, tag)] if r['seed'] == seed)


# ================================== Helpers ================================= #


def time_line(ax, xy, dt, norm, lw=2.0, dash_on=5, dash_off=3):
    """Dashed viridis trail as _plot_colour_line in disp_het_multi_rob.py, but colored by absolute
    time (norm) so that runs of different length share one color scale.

    Each dash is one polyline (colored by its start time) and dashes of a robot standing still are
    skipped, which keeps the pgf output small enough for TeX.
    """
    period = dash_on + dash_off
    dashes, starts = [], []
    for k in range(0, len(xy) - 1, period):
        dash = xy[k : k + dash_on + 1]
        if np.linalg.norm(dash[-1] - dash[0]) > 1e-3:
            dashes.append(dash)
            starts.append(k * dt)
    ax.add_collection(
        LineCollection(dashes, colors=colormaps['viridis'](norm(np.array(starts))), linewidths=lw)
    )


def method_handles(alpha=0.6):
    return [
        Patch(facecolor=c, edgecolor='k', lw=0.6, alpha=alpha, label=lab)
        for _, lab, c, _ in METHODS
    ]


def style_box_axes(ax):
    ax.grid(axis='x', visible=False)
    ax.grid(axis='y', linestyle='-.', alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def grouped_boxes(ax, groups, values, ticklabels, width=0.84, marker_values=None):
    """One box per method inside each group; values[g][m] is a 1-D array (may be empty)."""
    n = len(METHODS)
    bw = width / n
    for m, (_, _, color, _) in enumerate(METHODS):
        for g in range(len(groups)):
            pos = g + (m - (n - 1) / 2) * bw
            v = np.asarray(values[g][m], dtype=float)
            v = v[~np.isnan(v)]
            if v.size == 0:
                ax.text(
                    pos,
                    0.5,
                    'DNF',
                    transform=ax.get_xaxis_transform(),
                    rotation=90,
                    ha='center',
                    va='center',
                    fontsize=11,
                    color='#555555',
                )
                continue
            bp = ax.boxplot(
                [v],
                positions=[pos],
                widths=bw * 0.9,
                patch_artist=True,
                medianprops={'color': 'black'},
                flierprops={'markersize': 4},
            )
            for box in bp['boxes']:
                box.set_facecolor(color)
                box.set_alpha(0.6)
            if marker_values is not None and marker_values[g][m] is not None:
                ax.plot(
                    pos,
                    marker_values[g][m],
                    marker='D',
                    ms=6,
                    mfc='white',
                    mec='black',
                    mew=1.0,
                    ls='',
                    zorder=5,
                )
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(ticklabels)
    ax.set_xlim(-0.5, len(groups) - 0.5)
    style_box_axes(ax)


# ================================= Figures ================================== #


def fig_kpi_boxplot(runs, textsize):
    fig, axes = plt.subplots(5, 1, figsize=(8.5, 14), sharex=True)
    n = len(METHODS)
    bw = 0.84 / n

    # Completion ratio: safe finishes solid, finished-but-unsafe hatched on top.
    ax = axes[0]
    for m, (tag, _, color, _) in enumerate(METHODS):
        for g, s in enumerate(SCEN_KEYS):
            rs = runs[(s, tag)]
            safe = sum(r['safe_success'] for r in rs) / len(rs)
            unsafe = sum(r['converged'] and not r['safe_success'] for r in rs) / len(rs)
            pos = g + (m - (n - 1) / 2) * bw
            ax.bar(pos, safe, width=bw * 0.9, color=color, alpha=0.6, edgecolor='k', lw=0.6)
            if unsafe:
                ax.bar(
                    pos,
                    unsafe,
                    bottom=safe,
                    width=bw * 0.9,
                    facecolor='white',
                    edgecolor=color,
                    hatch='////',
                    lw=0.8,
                )
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Completion ratio', fontsize=textsize - 2)
    style_box_axes(ax)

    def per(fn):
        return [[[fn(r) for r in runs[(s, tag)]] for tag, *_ in METHODS] for s in SCEN_KEYS]

    ttg = [
        [[r['norm_ttg'] for r in runs[(s, tag)] if r['converged']] for tag, *_ in METHODS]
        for s in SCEN_KEYS
    ]
    grouped_boxes(axes[1], SCEN_KEYS, ttg, [])
    axes[1].set_ylabel('Normalized\ntime-to-goal', fontsize=textsize - 2)
    axes[1].set_ylim(0, 1.0)

    grouped_boxes(
        axes[2], SCEN_KEYS, per(lambda r: collision_events(r['x'], D_SAFE - COLLISION_MARGIN)), []
    )
    axes[2].set_ylabel('Collision events', fontsize=textsize - 2)

    grouped_boxes(axes[3], SCEN_KEYS, per(lambda r: max_violation(r['x'])), [])
    axes[3].set_ylabel('Max violation [$m$]', fontsize=textsize - 2)

    qp_median = [
        [
            float(np.nanmedian([r['qp_ms'] for r in runs[(s, tag)]]))
            if tag.startswith('dhqp')
            else None
            for tag, *_ in METHODS
        ]
        for s in SCEN_KEYS
    ]
    grouped_boxes(
        axes[4],
        SCEN_KEYS,
        per(lambda r: r['ms']),
        [SCEN_LABEL[s] for s in SCEN_KEYS],
        marker_values=qp_median,
    )
    axes[4].set_yscale('log')
    axes[4].set_ylabel('Time per robot\nand step [$ms$]', fontsize=textsize - 2)

    for ax in axes:
        ax.tick_params(axis='x', length=0)
    handles = method_handles() + [
        Patch(facecolor='white', edgecolor='k', hatch='////', lw=0.6, label='Finished, unsafe'),
        Line2D([], [], marker='D', ms=6, mfc='white', mec='black', ls='', label='dHQP solve only'),
    ]
    fig.legend(
        handles=handles, loc='outside lower center', ncol=3, frameon=False, fontsize=textsize - 1
    )
    fig.savefig(OUT / 'kpi_boxplot.pdf', backend='pgf', bbox_inches='tight', pad_inches=0.0)
    plt.close(fig)


def outcome_text(r):
    if r['converged']:
        return 'goal reached'
    return 'deadlock' if r['deadlock'] else f'{CAP_S:.0f} $s$ cap'


def fig_trajectories(runs, scen, seed, textsize):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10.6), sharex=True, sharey=True)
    lim = 7.4
    norm = Normalize(0, CAP_S)
    for ax, (tag, label, _, _) in zip(axes.flat, METHODS):
        r = run_of(runs, scen, tag, seed)
        x, goals = r['x'], r['goals']
        for i in range(x.shape[1]):
            time_line(ax, x[:, i], r['dt'], norm)
        if scen == 'priority_conflict':
            ax.plot(x[-1, :2, 0], x[-1, :2, 1], color='k', lw=1.5, zorder=4)
        for i in range(x.shape[1]):
            ax.add_patch(
                Circle(
                    x[-1, i],
                    D_SAFE / 2,
                    facecolor='#D95319',
                    edgecolor='#D95319',
                    alpha=0.3,
                    lw=1.0,
                    zorder=3,
                )
            )
            ax.plot(*x[-1, i], marker='o', color='#D95319', ms=6, zorder=5)
            ax.plot(*goals[i], marker='x', color='k', ms=8, mew=1.8, ls='', zorder=6)
            ax.annotate(
                f'$\\mathcal{{T}}_{i + 1}$',
                goals[i],
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=textsize - 3,
                zorder=6,
            )
        t = r['last_step'] * r['dt']
        ax.text(
            0.97,
            0.03,
            f'$t = {t:.2f}\\,s$\n{outcome_text(r)}\n$d_\\mathrm{{min}} = {r["min_distance"]:.2f}\\,m$',
            transform=ax.transAxes,
            ha='right',
            va='bottom',
            fontsize=textsize - 2,
            bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.8, 'pad': 2},
        )
        ax.set_title(label)
        ax.set(
            xlim=(-lim, lim),
            ylim=(-lim, lim),
            aspect='equal',
            xticks=range(-6, 7, 3),
            yticks=range(-6, 7, 3),
        )
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap='viridis'), ax=axes, shrink=0.6, pad=0.01)
    cbar.set_label('Time [$s$]')
    for ax in axes[-1]:
        ax.set_xlabel('$x$ [$m$]')
    for ax in axes[:, 0]:
        ax.set_ylabel('$y$ [$m$]')
    handles = [
        Line2D([], [], marker='o', color='#D95319', ls='', label='Omnidirectional robot'),
        Patch(facecolor='#D95319', alpha=0.3, label='Safety disc ($d_\\mathrm{safe}/2$)'),
        Line2D([], [], marker='x', color='k', mew=1.8, ls='', label='Goal'),
    ]
    if scen == 'priority_conflict':
        handles.append(Line2D([], [], color='k', lw=1.5, label='Formation pair'))
    fig.legend(handles=handles, loc='outside lower center', ncol=len(handles), frameon=False)
    fig.savefig(
        OUT / f'trajectories_{scen}.pdf', backend='pgf', bbox_inches='tight', pad_inches=0.0
    )
    plt.close(fig)


def fig_min_distance_time(runs, seeds, textsize):
    fig = plt.figure(figsize=(8.5, 7.5))
    gs = gridspec.GridSpec(3, 2, width_ratios=[1, 0.42], figure=fig)
    fig.set_constrained_layout_pads(wspace=0.0, w_pad=0.02)
    axes = [fig.add_subplot(gs[k, 0]) for k in range(3)]
    for ax in axes[1:]:
        ax.sharex(axes[0])
    for ax, scen in zip(axes, SCEN_KEYS):
        ax.axhspan(0, D_SAFE, color='red', alpha=0.25, lw=0)
        ax.axhline(ENFORCED, color='k', lw=0.8, ls='--', alpha=0.6)
        for tag, label, color, ls in METHODS:
            r = run_of(runs, scen, tag, seeds[scen])
            d = pair_distances(r['x']).min(axis=0)
            ax.plot(np.arange(len(d)) * r['dt'], d, color=color, ls=ls, lw=1.5, label=label)
        ax.set_ylim(0, 4)
        ax.set_xlim(0, CAP_S)
        ax.set_ylabel(f'{SCEN_LABEL[scen]}\n[$m$]', fontsize=textsize - 1)
        if ax is not axes[-1]:
            plt.setp(ax.get_xticklabels(), visible=False)
    axes[-1].set_xlabel('Time [$s$]')
    leg = fig.add_subplot(gs[:, 1])
    leg.axis('off')
    handles, labels = axes[0].get_legend_handles_labels()
    handles += [
        Patch(facecolor='red', alpha=0.25, label='$< d_\\mathrm{safe}$'),
        Line2D([], [], color='k', lw=0.8, ls='--', alpha=0.6, label='Enforced'),
    ]
    leg.legend(
        handles=handles,
        loc='center left',
        frameon=False,
        fontsize=textsize - 1,
        title='Min. inter-robot dist.',
        title_fontsize=textsize - 1,
    )
    fig.savefig(OUT / 'min_distance_time.pdf', backend='pgf', bbox_inches='tight', pad_inches=0.0)
    plt.close(fig)


def fig_priority_conflict_tasks(runs, seed, textsize):
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5), sharex=True, sharey=True)
    for ax, (tag, label, _, _) in zip(axes.flat, METHODS):
        r = run_of(runs, 'priority_conflict', tag, seed)
        x, goals = r['x'], r['goals']
        t = np.arange(len(x)) * r['dt']
        g, f = ROLES[tag]
        ax.axhspan(D_FORM - FORM_TOL, D_FORM + FORM_TOL, color='#77AC30', alpha=0.35, lw=0)
        ax.axhline(D_FORM, color='#77AC30', lw=1.0, ls='--')
        ax.plot(t, np.linalg.norm(x[:, 0] - x[:, 1], axis=1), color='#0072BD', ls='-', lw=1.8)
        ax.plot(t, np.linalg.norm(x[:, g] - goals[g], axis=1), color='#D95319', ls='--', lw=1.8)
        ax.plot(t, np.linalg.norm(x[:, f] - goals[f], axis=1), color='#EDB120', ls='-.', lw=1.8)
        ax.set_title(label)
        ax.set(xlim=(0, CAP_S), ylim=(0, 12.5))
    for ax in axes[-1]:
        ax.set_xlabel('Time [$s$]')
    for ax in axes[:, 0]:
        ax.set_ylabel('Distance [$m$]')
    handles = [
        Line2D([], [], color='#0072BD', ls='-', lw=1.8, label='Pair distance'),
        Patch(facecolor='#77AC30', alpha=0.35, label='$d_\\mathrm{form} \\pm 5\\,cm$'),
        Line2D([], [], color='#D95319', ls='--', lw=1.8, label='Goal error, goal-first robot'),
        Line2D([], [], color='#EDB120', ls='-.', lw=1.8, label='Goal error, formation-first robot'),
    ]
    fig.legend(handles=handles, loc='outside lower center', ncol=4, frameon=False)
    fig.savefig(
        OUT / 'priority_conflict_tasks.pdf', backend='pgf', bbox_inches='tight', pad_inches=0.0
    )
    plt.close(fig)


def fig_priority_conflict_final_errors(runs, textsize):
    groups = ['form', 'goal_first', 'form_first']
    values = [[], [], []]
    for tag, *_ in METHODS:
        g, f = ROLES[tag]
        rs = runs[('priority_conflict', tag)]
        fin = [r['x'][-1] for r in rs]
        goals = [r['goals'] for r in rs]
        values[0].append([abs(np.linalg.norm(p[0] - p[1]) - D_FORM) for p in fin])
        values[1].append([np.linalg.norm(p[g] - q[g]) for p, q in zip(fin, goals)])
        values[2].append([np.linalg.norm(p[f] - q[f]) for p, q in zip(fin, goals)])
    values = [[np.maximum(v, 1e-4) for v in grp] for grp in values]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    grouped_boxes(
        ax,
        groups,
        values,
        ['Formation error', 'Goal error,\ngoal-first robot', 'Goal error,\nformation-first robot'],
    )
    ax.set_yscale('log')
    ax.set_ylabel('Final error [$m$]')
    for g, tol in enumerate((FORM_TOL, GOAL_TOL, GOAL_TOL)):
        ax.hlines(tol, g - 0.45, g + 0.45, color='k', lw=1.0, ls='--')
    ax.tick_params(axis='x', length=0)
    handles = method_handles() + [Line2D([], [], color='k', lw=1.0, ls='--', label='Tolerance')]
    fig.legend(
        handles=handles, loc='outside lower center', ncol=4, frameon=False, fontsize=textsize - 1
    )
    fig.savefig(
        OUT / 'priority_conflict_final_errors.pdf',
        backend='pgf',
        bbox_inches='tight',
        pad_inches=0.0,
    )
    plt.close(fig)


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        '--seed', type=int, default=None, help='representative seed for every scenario'
    )
    parser.add_argument(
        '--sources',
        type=Path,
        nargs='+',
        default=SOURCES,
        help='campaign output directories, each with analysis/ and <scenario>/<tag>/<seed>/',
    )
    parser.add_argument(
        '--out-dir', type=Path, default=OUT, help='directory that receives the PDFs'
    )
    args = parser.parse_args()

    OUT = args.out_dir
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcdefaults()
    textsize = init_matplotlib()
    # Matplotlib 3.11's usetex PDF writer drops the CMSY minus (glyph 0), so figures are saved
    # through pgf/pdflatex with the preamble usetex would use (same fonts as the paper's figures).
    plt.rcParams.update(
        {
            'axes.unicode_minus': False,
            'pgf.texsystem': 'pdflatex',
            'pgf.rcfonts': False,
            'pgf.preamble': r'\usepackage{mathptmx} ' + plt.rcParams['text.latex.preamble'],
        }
    )

    runs = load_runs(args.sources)
    seeds = representative_seeds(runs, args.seed)
    print('representative seeds:', seeds)
    for scen in SCEN_KEYS:
        for tag, label, *_ in METHODS:
            rs = runs[(scen, tag)]
            ev = [collision_events(r['x'], D_SAFE - COLLISION_MARGIN) for r in rs]
            print(
                f'{scen:18s} {label:20s} safe {sum(r["safe_success"] for r in rs):2d}/{len(rs)} '
                f'finished {sum(r["converged"] for r in rs):2d}/{len(rs)} collision events {sum(ev)}'
            )

    fig_kpi_boxplot(runs, textsize)
    for scen in SCEN_KEYS:
        fig_trajectories(runs, scen, seeds[scen], textsize)
    fig_min_distance_time(runs, seeds, textsize)
    fig_priority_conflict_tasks(runs, seeds['priority_conflict'], textsize)
    fig_priority_conflict_final_errors(runs, textsize)
    print('wrote', sorted(p.name for p in OUT.glob('*.pdf')))


if __name__ == '__main__':
    main()
