# Comparison: dHQP vs. centralized HQP vs. three lightweight baselines

## Omnidirectional benchmark (colleague's three scenarios)

`run_comparison.py --preset colleague_omni` runs the benchmark of `origin/dhqp_with_plots` (6 omnidirectional agents, radius 6, dt 0.03, 1200 steps, v_max 1.4, d_safe 1.5) on the scenarios `uniform`, `asymmetric` and `priority_conflict`, with his simulators from `potential_field/`, `distributed_qp/`, `cbf_qp/`, `orca_radial_switching/`, `nh_orca_radial_switching/` and `dhqp_radial_switching/` driven through `methods/omni_adapters.py`.
Seed k is his instance k (`common/benchmark.py:generate_omni_instance`).

Rules every method shares:

- every constraint-based method enforces `d_safe + 2*v_max*dt` (1.584 m), and all are scored against `d_safe`;
- every integrator clips `||u|| <= v_max`;
- a run stops, and is scored as converged, on the same rule: all agents within 1 cm of their goals, except the formation pair in `priority_conflict` for formation-capable methods, judged on the pair distance within 5 cm of `d_form`;
- timing is `perf_counter` around each agent's control law, one job per process, one job per physical core (`--workers`, `--pin-cpus`).

The NH-ORCA folder has the same `orca.py` and `node.py` as the ORCA one; its only difference was a larger radius, so under the shared enforced distance the two are the same controller.

Full campaign, inside the `ho_mpc` container from the workspace root:

```bash
source install/setup.bash
src/distributed_ho_mpc/distributed_ho_mpc/scenarios/comparison/run_omni_campaign.sh out/<campaign> 0:15
```

`analyze_comparison.py out/<campaign>` (run by the script) writes `summary_<scenario>.csv`, `summary_by_method.csv` and `table_colleague_format.{md,tex}`.

## What this is

This scenario runs **every distributed-control method implemented in this
repo** on the literal same radial-switching benchmark and produces a single
KPI table plus overlay plots, aimed at answering the reviewer question this
was built for: *if I need to do formation control (or radial switching, ...),
should I use dHQP, or a simpler/cheaper alternative -- and why?*

Methods compared:

| Method | Folder | What it is |
|---|---|---|
| dHQP | `dhqp_radial_switching/` | This repo's contribution: distributed, per-agent hierarchical QP with consensus, exact lexicographic task priorities, priority order settable per agent |
| Centralized HQP | `centralized_radial_switching/` | The paper's existing baseline: one global hierarchical QP over the whole fleet, exact priorities but only ONE shared order for everyone |
| Weighted distributed QP | `distributed_qp/` | Distributed, per-agent QP, hard safety via a CBF constraint, but goal/formation are soft weighted costs in ONE QP -- no priority |
| ORCA | `orca_radial_switching/` | Distributed reciprocal velocity-obstacle collision avoidance (van den Berg et al., 2011) + go-to-goal; no formation/coupling task at all |
| Potential field | `potential_field/` | Distributed heuristic force sum (goal attraction + collision repulsion + optional formation spring), no optimization, no priority |

Two scenarios are run for each method:

- **`uniform`**: every agent only has a goal-reaching task (protected by
  safety/collision-avoidance). This is the case a lightweight baseline is
  designed for.
- **`priority_conflict`**: agents 0 and 1 must additionally hold a formation
  with each other, and the two agents disagree about which matters more
  (formation vs. their own goal). dHQP expresses this as a genuinely
  different LOCAL task hierarchy per agent (no retuning: only the `prio`
  integers change, see `dhqp_radial_switching/network_simulation.py:
  build_system_tasks`); every other method can only *bias* the outcome via
  gains/weights, and centralized HQP cannot even express a per-agent
  difference (a single global hierarchy only).

## Why the benchmark is aligned across methods

Each method's own `settings.py` has its own defaults, tuned for developing
that method in isolation. `run_comparison.py` monkey-patches every method's
settings module to the **same** canonical values from this folder's
`settings.py` (`n_nodes`, `radius`, `dt`, `n_steps`, `v_max`,
`communication_range`, `d_safe`, `d_form`, `goal_tol`,
`formation_pairs`) before calling its `run()`, so all five methods start
from the same 5-agent circle layout, chase the same antipodal goals, share
the same speed limit and safety/formation distances, and are judged
convergent by the same 1 cm tolerance (matching the paper's own
"normalized time-to-goal" definition).

## KPIs (`kpi_table.csv` / `kpi_table.md`)

Deliberately **not** led by raw performance, per the reviewer's own framing
("focus more not on performances but on task execution"):

- `converged`, `time_to_goal_s`, `normalized_time_to_goal` -- did the method
  reach every goal, and how fast (only meaningful when converged).
- `min_distance_m` / `safety_margin_ok` -- worst-case inter-robot distance
  over the whole run vs. `d_safe`.
- `formation_pair_distance_m` / `formation_error_m` -- final distance
  between agents 0-1 vs. the `d_form` target -- the direct measure of
  whether the "dominant" task in `priority_conflict` was actually honored.
- `goal_error_agent_a_m` / `goal_error_agent_b_m` -- final distance to goal
  for agents 0 and 1 -- shows which agent "won" the conflict, and by how
  much the other was sacrificed.
- `wall_time_s` / `solve_time_s` -- compute cost, reported for completeness
  but captioned as secondary, not the headline.
- `strict_priority`, `per_agent_priority`, `formation_tasks`,
  `retuning_to_change_priority` -- qualitative, not measured by running a
  simulation: whether the control law can express a strict priority order
  at all, whether that order can differ per agent, whether it can express a
  formation/coupling task at all, and what changing the scenario costs. This
  is the actual crux of the argument this comparison exists to make.

## Known issue: dHQP currently fails to run

As of this comparison, `dhqp_radial_switching` (and the pre-existing
`radial_switching` scenario it reuses `Node` from) fails inside
`distributed_ho_mpc/ho_mpc/ho_mpc_multi_robot_copy.py`. Confirmed
independent of anything in this comparison folder -- it reproduces on the
pristine, unmodified `radial_switching` scenario too:

- `ho_mpc/ho_mpc_multi_robot_copy.py:__call__` (around line 1497) returns
  **4** values: `return u_0, s, lamb_P, w_P`.
- `radial_switching/node.py:484` (used by `dhqp_radial_switching` too) only
  unpacks **2**: `self.u_star, self.y = self.hompc(...)`.
- Separately, `Node.dual_update()` (`radial_switching/node.py`, around line
  591-596) returns immediately after `self.save_data()` -- the numeric
  `rho_i` consensus update below that `return` is unreachable dead code, so
  `rho_i` never actually updates from its zero initialization.

Both point to the distributed consensus solver being mid-refactor in this
checkout, not to anything specific to this benchmark. `run_comparison.py`
catches the exception per-method so the rest of the comparison still runs;
the dHQP row is reported with `status: ERROR` and the qualitative columns
still filled in from `CAPABILITIES` (since those are a property of the
method's design, not of whether the current checkout runs).

`dhqp_radial_switching/` itself is otherwise complete and ready to run: same
generic `Node`/task-dict machinery as `radial_switching`, same benchmark
layout as the other four methods (`build_radial_configuration`), per-agent
priority reordering already wired up in `build_system_tasks()`. It should
be re-run once the solver bug above is fixed.

## Run it

```shell
python3 src/distributed_ho_mpc/distributed_ho_mpc/scenarios/comparison/run_comparison.py
```

Writes `kpi_table.csv`, `kpi_table.md`, and `overlay_uniform.pdf` /
`overlay_priority_conflict.pdf` (agents 0-1 distance over time, every method
on one axes) to a timestamped folder under `out/`.

## Parameters that must be swept before reporting

Per-method parameters are passed with `--params`, keyed by method name or by
a `method@tag` tag (the tag also names the output subfolder), so the same
method can be run several times at different settings in one campaign:

```shell
--params '{"orca@slow": {"v_track_max": 0.8}, "orca@fast": {"v_track_max": 1.2}}'
```

Two parameters currently decide headline numbers and should not be left at
their defaults in a reported campaign:

- **`orca` / `v_track_max`** (default `min(v_h_max, epsilon * k_omega)` =
  0.8 m/s). This is the radius of NH-ORCA's trackable-velocity disc. The
  default derives from a worst-case `sin(e) = 1` lateral-drift bound that
  partly double-counts the error already absorbed by the `2 * epsilon`
  radius inflation, and it sits below the ~0.9 m/s needed to cross this
  benchmark within `max_steps` -- so it sets the NH-ORCA success rate
  outright. See the `Caveat on the default cap` section of
  `methods/nh_orca.py`.
- **`cbf` / `gamma`** (default 1.0). The class-K function is now linear;
  under the previous cubic `h**3` a gamma sweep returned byte-identical
  results, so any gamma conclusion predating that change is void and must
  be re-run.

## Safety contract each method actually enforces

The methods do not all enforce the same center-to-center distance, so each
run records `enforced_safety_distance` in its `run_info.json` metadata and
the KPI table should be read next to it:

| Method | Enforced | Reason |
| --- | --- | --- |
| `cbf` | `safety_distance + 2*l` (2.6 by default) | Barrier guards the feedback-linearization offset points, not the centers |
| `orca` | `safety_distance + 2*epsilon` (2.4 by default) | ORCA radius inflated to absorb holonomic tracking error |
| `dhqp` | `safety_distance + margin` (2.0 by default) | Enforced exactly; `margin` exists to grant the same latitude if a campaign wants it |
