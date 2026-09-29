"""Figures (PDF) and tables for the omnidirectional radial-switching comparison.

Reads the per-run summaries written by analyze_comparison.py, by default from the workspace out/:
  out/omni_campaign_2026-09-25/analysis/summary_<scenario>.csv   (campaign, neighbor_limit 2)
  out/omni_nl5_2026-09-26/analysis/summary_<scenario>.csv        (dHQP with every pair linked)
and writes figures/*.pdf and tables/*.{csv,md,tex} to out/omni_results/.

usage: python3 make_results.py [--sources DIR [DIR ...]] [--out-dir DIR]
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SOURCES = [ROOT / 'out' / 'omni_campaign_2026-09-25', ROOT / 'out' / 'omni_nl5_2026-09-26']
OUT_DIR = ROOT / 'out' / 'omni_results'
FIG = OUT_DIR / 'figures'
TAB = OUT_DIR / 'tables'

SCENARIOS = [
    ('uniform', 'Uniform'),
    ('asymmetric', 'Asymmetric'),
    ('priority_conflict', 'Priority conflict'),
]
# Figure rows. NH-ORCA is omitted: same code as ORCA, identical on every run.
METHODS = [
    ('dhqp_omni@nl5', 'dHQP, all pairs linked', True),
    ('dhqp_omni', 'dHQP, 2 nearest linked', True),
    ('pf', 'Potential field', False),
    ('dqp', 'Distributed QP', False),
    ('cbf_omni', 'CBF-QP', False),
    ('orca_omni', 'ORCA (= NH-ORCA)', False),
]
TABLE_METHODS = METHODS[:2] + [('dhqp_omni@nc4', 'dHQP, 2 nearest, $n_c$ = 4', True)] + METHODS[2:]

D_SAFE, ENFORCED, CAP_S = 1.5, 1.584, 36.0

# Palette: validated reference slots 1-2 on the reference light surface.
SURFACE = '#fcfcfb'
INK, INK2, MUTED = '#0b0b0b', '#52514e', '#898781'
GRID, AXIS, BAND = '#e1e0d9', '#c3c2b7', '#f0efec'
OUT_COLOR = {'safe': '#2a78d6', 'unsafe': '#eb6834', 'dnf': '#d6d5ce'}
OUT_LABEL = {
    'safe': 'Safe finish',
    'unsafe': 'Finished, closer than 1.5 m at some point',
    'dnf': 'Did not finish in 1,200 steps (36 s)',
}

plt.rcParams.update(
    {
        'font.family': 'DejaVu Sans',
        'font.size': 8.5,
        'axes.edgecolor': AXIS,
        'axes.labelcolor': INK2,
        'axes.titlesize': 9.5,
        'axes.titleweight': 'bold',
        'axes.titlecolor': INK,
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


def load_runs(sources):
    runs = []
    for src in sources:
        for key, _ in SCENARIOS:
            with open(src / 'analysis' / f'summary_{key}.csv') as f:
                for r in csv.DictReader(f):

                    def num(k):
                        return float('nan') if r[k] in ('', 'nan') else float(r[k])

                    runs.append(
                        {
                            'scenario': key,
                            'tag': r['method_tag'],
                            'seed': int(r['seed']),
                            'converged': r['converged'] == 'True',
                            'safe_success': r['safe_success'] == 'True',
                            'deadlock': r['deadlock'] == 'True',
                            'norm_ttg': num('norm_ttg'),
                            'min_distance': num('min_distance'),
                            'ms_per_agent_step': num('ms_per_agent_step'),
                            'qp_ms_per_agent_step': num('qp_ms_per_agent_step'),
                            'wall_time_s': num('wall_time_s'),
                            'last_step': int(r['last_step']),
                        }
                    )
    return runs


def outcome(r):
    return 'safe' if r['safe_success'] else 'unsafe' if r['converged'] else 'dnf'


def select(runs, scenario, tag):
    return sorted(
        (r for r in runs if r['scenario'] == scenario and r['tag'] == tag), key=lambda r: r['seed']
    )


def style_rows(ax, n):
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_yticks(range(n))
    ax.set_yticklabels([m[1] for m in METHODS])
    for lbl, (_, _, em) in zip(ax.get_yticklabels(), METHODS):
        lbl.set_fontweight('bold' if em else 'normal')
    for i, (_, _, em) in enumerate(METHODS):
        if em:
            ax.axhspan(i - 0.5, i + 0.5, color=BAND, zorder=0, lw=0)
    ax.tick_params(axis='y', length=0)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.grid(axis='x', color=GRID, lw=0.6, zorder=0.5)
    ax.set_axisbelow(True)


def fig_outcomes(runs):
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 2.9), sharey=True, constrained_layout=True)
    for ax, (key, label) in zip(axes, SCENARIOS):
        for i, (tag, _, _) in enumerate(METHODS):
            rs = select(runs, key, tag)
            left = 0
            for k in ('safe', 'unsafe', 'dnf'):
                n = sum(outcome(r) == k for r in rs)
                if n:
                    ax.barh(
                        i,
                        n,
                        left=left,
                        height=0.52,
                        color=OUT_COLOR[k],
                        edgecolor=SURFACE,
                        lw=1.5,
                        zorder=2,
                    )
                    left += n
            ax.text(
                15.4,
                i,
                f'{sum(r["safe_success"] for r in rs)}/15',
                va='center',
                ha='left',
                color=INK,
                fontsize=8,
            )
        style_rows(ax, len(METHODS))
        ax.set_xlim(0, 15)
        ax.set_xticks([0, 5, 10, 15])
        ax.set_title(label, loc='left')
        ax.set_xlabel('runs (of 15)')
    handles = [Patch(color=OUT_COLOR[k], label=OUT_LABEL[k]) for k in ('safe', 'unsafe', 'dnf')]
    fig.legend(handles=handles, loc='outside lower center', ncol=3, frameon=False, fontsize=8)
    fig.savefig(FIG / 'outcomes.pdf')
    plt.close(fig)


def fig_strip(
    runs,
    name,
    value,
    include,
    xlabel,
    xlim,
    log=False,
    refs=(),
    color=None,
    empty='none finished',
    legend=('safe', 'unsafe', 'dnf'),
):
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 2.9), sharey=True, constrained_layout=True)
    for ax, (key, label) in zip(axes, SCENARIOS):
        for i, (tag, _, _) in enumerate(METHODS):
            rs = [r for r in select(runs, key, tag) if include(r)]
            if not rs:
                ax.text(xlim[0], i, f'  {empty}', va='center', ha='left', color=MUTED, fontsize=7.5)
                continue
            for r in rs:
                jitter = ((r['seed'] * 7) % 5 - 2) * 0.07
                c = color(r) if color else OUT_COLOR[outcome(r)]
                v = min(max(value(r), xlim[0]), xlim[1])
                ax.scatter(
                    v, i + jitter, s=22, color=c, edgecolors=SURFACE, linewidths=1.0, zorder=3
                )
        for v, lab in refs:
            ax.axvline(v, color=INK2, lw=0.8, zorder=2)
            ax.text(
                v,
                -0.62,
                lab,
                rotation=0,
                ha='right' if v < 1.55 else 'left',
                va='bottom',
                fontsize=7,
                color=INK2,
            )
        style_rows(ax, len(METHODS))
        if log:
            ax.set_xscale('log')
            ax.set_xticks([0.01, 0.1, 1, 10])
            ax.set_xticklabels(['0.01', '0.1', '1', '10'])
            ax.minorticks_off()
        ax.set_xlim(*xlim)
        ax.set_title(label, loc='left', pad=14 if refs else 6)
    fig.supxlabel(xlabel, color=INK2, fontsize=8.5)
    if color is None:
        handles = [
            plt.Line2D(
                [], [], ls='', marker='o', ms=5, mfc=OUT_COLOR[k], mec=SURFACE, label=OUT_LABEL[k]
            )
            for k in legend
        ]
        fig.legend(handles=handles, loc='outside lower center', ncol=3, frameon=False, fontsize=8)
    fig.savefig(FIG / name)
    plt.close(fig)


def mean_sd(values):
    values = [v for v in values if not np.isnan(v)]
    if not values:
        return None
    return float(np.mean(values)), float(np.std(values))


def summarize(runs):
    rows = []
    for key, slabel in SCENARIOS:
        for tag, mlabel, _ in TABLE_METHODS:
            rs = select(runs, key, tag)
            fin = [r for r in rs if r['converged']]
            ttg = mean_sd([r['norm_ttg'] for r in fin])
            md = mean_sd([r['min_distance'] for r in rs])
            rows.append(
                {
                    'scenario': slabel,
                    'method': mlabel.replace('$n_c$', 'n_c'),
                    'method_tex': mlabel,
                    'tag': tag,
                    'runs': len(rs),
                    'finished': len(fin),
                    'safe_finish': sum(r['safe_success'] for r in rs),
                    'deadlock': sum(r['deadlock'] for r in rs),
                    'norm_ttg_mean': ttg[0] if ttg else None,
                    'norm_ttg_sd': ttg[1] if ttg else None,
                    'min_dist_mean': md[0],
                    'min_dist_sd': md[1],
                    'min_dist_worst': min(r['min_distance'] for r in rs),
                    'wall_s_mean': float(np.mean([r['wall_time_s'] for r in rs])),
                    'ms_per_robot_step': float(np.mean([r['ms_per_agent_step'] for r in rs])),
                    'qp_ms_per_robot_step': mean_sd([r['qp_ms_per_agent_step'] for r in rs]),
                }
            )
    return rows


def write_tables(runs, rows):
    with open(TAB / 'runs.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(runs[0].keys()))
        w.writeheader()
        w.writerows(sorted(runs, key=lambda r: (r['scenario'], r['tag'], r['seed'])))

    fields = [k for k in rows[0] if k not in ('method_tex', 'qp_ms_per_robot_step')] + [
        'qp_ms_per_robot_step'
    ]
    with open(TAB / 'summary.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    **r,
                    'qp_ms_per_robot_step': r['qp_ms_per_robot_step'][0]
                    if r['qp_ms_per_robot_step']
                    else '',
                }
            )

    def cells(r, pm):
        ttg = (
            f'{r["norm_ttg_mean"]:.3f}{pm}{r["norm_ttg_sd"]:.3f}'
            if r['norm_ttg_mean'] is not None
            else 'DNF'
        )
        ms = r['ms_per_robot_step']
        qp = r['qp_ms_per_robot_step']
        cost = f'{ms:.3f}' if ms < 1 else f'{ms:.2f}'
        if qp:
            cost += f' ({qp[0]:.2f})'
        return [
            f'{r["finished"]}/{r["runs"]}',
            f'{r["safe_finish"]}/{r["runs"]}',
            f'{r["deadlock"]}/{r["runs"]}',
            ttg,
            f'{r["min_dist_mean"]:.3f}{pm}{r["min_dist_sd"]:.3f}',
            f'{r["min_dist_worst"]:.3f}',
            cost,
        ]

    head = [
        'Finished',
        'Safe finish',
        'Deadlock',
        'Norm. TTG',
        'Min. dist [m]',
        'Worst [m]',
        'ms/robot-step (QP)',
    ]
    md = ['| Scenario | Method | ' + ' | '.join(head) + ' |', '|---|---|' + '---|' * len(head)]
    for r in rows:
        md.append(f'| {r["scenario"]} | {r["method"]} | ' + ' | '.join(cells(r, ' ± ')) + ' |')
    md.append('')
    md.append(
        'Safe finish: every robot within 1 cm of its goal (priority conflict, formation-capable methods: '
        'pair 0-1 within 5 cm of 2 m instead) and minimum distance >= d_safe = 1.5 m. '
        'Norm. TTG: steps to finish / 1200, finished runs only. Min. dist: mean ± sd over 15 seeds; '
        'Worst: smallest over the 15. ms/robot-step: mean per-robot compute time of one control step, '
        'HQP-only share in parentheses. Every constraint-based method enforces 1.584 m. '
        'dHQP all pairs linked: separate run, 2026-09-26, same image, pinning and seeds.'
    )
    (TAB / 'summary.md').write_text('\n'.join(md) + '\n')

    tex = [
        '% Requires \\usepackage{booktabs}',
        '\\begin{tabular}{llrrrrrrr}',
        '\\toprule',
        'Scen. & Method & Fin. & Safe & Deadl. & Norm. TTG & Min. dist [m] & Worst [m] & ms/step \\\\',
        '\\midrule',
    ]
    for key, slabel in SCENARIOS:
        block = [r for r in rows if r['scenario'] == slabel]
        for j, r in enumerate(block):
            c = [x.replace('DNF', '--') for x in cells(r, '$\\pm$')]
            name = (
                r['method_tex'].replace('=', '$=$')
                if '$' not in r['method_tex']
                else r['method_tex']
            )
            if r['tag'].startswith('dhqp'):
                name = f'\\textbf{{{name}}}'
            tex.append(f'{slabel if j == 0 else ""} & {name} & ' + ' & '.join(c) + ' \\\\')
        tex.append('\\midrule' if key != SCENARIOS[-1][0] else '\\bottomrule')
    tex.append('\\end{tabular}')
    (TAB / 'summary.tex').write_text('\n'.join(tex) + '\n')


def main():
    global FIG, TAB
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        '--sources',
        type=Path,
        nargs='+',
        default=SOURCES,
        help='campaign output directories, each with analysis/summary_<scenario>.csv',
    )
    parser.add_argument(
        '--out-dir', type=Path, default=OUT_DIR, help='directory that receives figures/ and tables/'
    )
    args = parser.parse_args()

    FIG, TAB = args.out_dir / 'figures', args.out_dir / 'tables'
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    runs = [r for r in load_runs(args.sources) if r['tag'] != 'nh_orca_omni']

    fig_outcomes(runs)
    fig_strip(
        runs,
        'time_to_goal.pdf',
        value=lambda r: r['norm_ttg'],
        include=lambda r: r['converged'],
        xlabel='steps to finish / 1,200 step cap (finished runs only)',
        xlim=(0, 1),
        legend=('safe', 'unsafe'),
    )
    fig_strip(
        runs,
        'min_distance.pdf',
        value=lambda r: r['min_distance'],
        include=lambda r: True,
        xlabel='closest approach between any two robots [m]',
        xlim=(0.5, 1.7),
        refs=[(D_SAFE, '$d_{safe}$ 1.5 '), (ENFORCED, ' enforced 1.584')],
    )
    fig_strip(
        runs,
        'compute.pdf',
        value=lambda r: r['ms_per_agent_step'],
        include=lambda r: True,
        xlabel='compute per robot per control step [ms, log scale]',
        xlim=(0.005, 20),
        log=True,
        color=lambda r: INK2,
    )
    write_tables(runs, summarize(runs))
    print(
        f'wrote {sorted(p.name for p in FIG.iterdir())} and {sorted(p.name for p in TAB.iterdir())}'
    )


if __name__ == '__main__':
    main()
