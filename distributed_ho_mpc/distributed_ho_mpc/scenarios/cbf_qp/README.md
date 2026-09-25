# CBF-QP scenario (weighted, non-hierarchical)

## What this is

A **distributed, per-agent weighted QP safety-filter controller**, using a
**control barrier function (CBF)** for its hard safety constraint. It plays
the same "lightweight baseline" role as
[`potential_field`](../potential_field/README.md), but replaces the
heuristic weighted potential-field law with an actual **constrained
optimization solved independently by each agent** — a single,
**non-hierarchical** QP that hard-enforces safety and softly blends
competing objectives by weight. It uses the repo's core
`hierarchical_qp.HierarchicalQP` solver in its `hierarchical=False`
("weighted") mode, not dHQP's full distributed-consensus machinery
(`radial_switching`). This lives entirely in this folder:

- `node.py` — defines the `Agent` class (the control law). Not an `rclpy`
  ROS 2 node — a plain Python class with no ROS dependency.
- `settings.py` — scenario parameters (QP solver, gains, weights, layout,
  scenario variant).
- `network_simulation.py` — the runnable script (`main()`), no ROS required.

Run it directly with:
```shell
python3 src/distributed_ho_mpc/distributed_ho_mpc/scenarios/cbf_qp/network_simulation.py
```
There is no `console_scripts` entry point or launch file for it — it's a
standalone offline simulation, like the other scenarios under
`distributed_ho_mpc/scenarios/`.

## Is it "fully distributed"?

**The control law is distributed; the simulation harness is centralized/sequential** —
the same split as `potential_field` and dHQP's `radial_switching`.

- Each `Agent.compute_input(...)` builds and solves **its own small QP**,
  using only its own state (`pos`, `goal`) and a dict of neighbor positions
  **filtered by `communication_range`**. There is no shared/joint
  optimization problem, no central solver, and no dual-variable/consensus
  exchange between agents (unlike `radial_switching`'s dHQP, which
  explicitly passes `rho_i`/`rho_j` messages between neighbors every inner
  loop). Each agent's QP is entirely local.
- `network_simulation.py` still runs everything in a single Python process:
  it builds one global `positions` dict from all agents every step and hands
  it to each `Agent.compute_input`, rather than simulating actual message
  passing / bandwidth-limited communication:

  ```python
  for step in range(st.n_steps):
      positions = {a.node_id: a.pos for a in agents}
      inputs = [a.compute_input(positions, st.communication_range) for a in agents]
      for a, u in zip(agents, inputs):
          a.step(u, st.dt)
  ```

## How the control law works: ONE weighted QP, not a priority cascade

At every step, agent `i` solves a **single QP** over its own commanded
velocity `u = [vx, vy]` (`Agent.compute_input` in `node.py`). It calls
`HierarchicalQP(..., hierarchical=False)`, which merges every task it is
given into **one combined QP solve** — there is no lexicographic ordering,
only per-task weights that decide how much each one counts in the shared
cost/constraint set:

1. **Velocity limits + collision avoidance -- heavily-penalized SOFT
   (`wi = w_safety`, default `1e3`).**
   A box constraint `-v_max <= vx, vy <= v_max`, plus a discrete-time
   **control barrier function (CBF)** inequality per neighbor within
   `communication_range`:

   ```python
   d_safe_enforced = d_safe + 2 * l
   h = dist**2 - d_safe_enforced**2
   C_row = -2.0 * (pos_i - pos_j)
   d_row = gamma * h
   # enforced as: C_row @ u <= d_row
   ```

   `wi = w_safety` (a large finite number, not `np.inf`) tells the solver's
   weighted mode to add a heavily-penalized slack variable to this block
   instead of enforcing it exactly with none (see
   `HierarchicalQP._solve_weighted`). See "Why the safety constraint is
   softened, not hard" below.

   **Why `d_safe_enforced` and not `d_safe` directly**: `d_safe` is the
   shared measurement threshold every method in `comparison/` is scored
   against; `l` is an extra conservative buffer (agent footprint/inflation)
   this baseline holds internally, so the CBF genuinely enforces more
   separation than what's measured — the same "enforced > measured"
   asymmetry a real CBF-QP controller needs for robustness/feasibility
   margin, rather than shaving the barrier down to exactly the scored
   distance (which leaves no recovery margin if `gamma`/`v_max` can't react
   instantly).
2. **Goal tracking -- SOFT (`we = w_goal`).**
   `u = k_goal * (goal - pos)` as an equality task, weighted by `w_goal` in
   the combined least-squares cost.
3. **Formation keeping -- SOFT (`we = w_form`)**, only when the agent has
   `formation_targets`: `direction . u = -k_form * (dist - d_form)`, weighted
   by `w_form`.

Because tasks 2 and 3 are both **soft, weighted equality costs solved
together in the same QP**, there is **no strict priority** between them —
exactly like `potential_field`'s weighted vector sum, except here the
trade-off is the exact solution of a weighted-least-squares problem (subject
to the hard safety constraints), not a raw heuristic force addition. Compare
this to a genuinely **hierarchical** distributed-QP variant (cascaded
priority levels, `hierarchical=True`), where the higher-priority task would
be satisfied *exactly* first and the lower-priority one only optimized in
the leftover null space — deliberately not what this scenario does.

Position integration is explicit Euler: `pos += dt * u` (`Agent.step`).

## Why the safety constraint is softened, not hard

An earlier version of this scenario used `wi = np.inf` for Level 0 (a
genuine hard constraint, no slack). That is the textbook CBF-QP guarantee,
but it means the *whole* per-agent QP raises `RuntimeError` the moment the
hard constraints (box + every active neighbor's CBF row) become mutually
infeasible at some step — which does happen under multi-agent congestion:
e.g. in the `priority_conflict` scenario with 6 agents crossing near the
centre, one agent can end up simultaneously pinned at exactly
`d_safe_enforced` against two different neighbors in conflicting
directions, at which point no `u` inside the velocity box satisfies both
CBF rows at once. This is a real, citable limitation of discrete-time
CBF-QP under congestion (the same family of finding as CBF-QP's
symmetric-encounter deadlock), not a bug in the margin/wiring — but a
`RuntimeError` mid-run kills that entire method/scenario cell instead of
reporting a degraded-but-informative result.

`w_safety` (default `1e3`) trades the "exact hard constraint" guarantee for
graceful degradation: the same block becomes a heavily-penalized soft cost,
satisfied to within ~1e-4 of the target margin whenever the exact hard
problem would have been feasible, but allowed a small, bounded violation
instead of raising when it wouldn't. Set `w_safety = np.inf` to restore the
exact hard-constraint version (and its `RuntimeError`-on-infeasibility
behavior).

**Keep `w_safety` well below ~1e4.** The slack block's diagonal in the QP's
cost matrix is `w_safety**2`, so `w_safety = 1e4` makes it `~1e8` against
`~1` for the goal-tracking block -- that conditioning gap is enough to make
`quadprog` spuriously return "no solution" for a QP that is, by
construction, always feasible (the slack variables are only lower-bounded,
never upper-bounded). Confirmed empirically on the `priority_conflict`
scenario: `w_safety` in `{1e2, 1e3, 3e3}` all run the full 1200-step
campaign cleanly with `min_distance` matching `d_safe + 2*l` to ~1e-4;
`1e4` fails partway through with the same `RuntimeError` the unsoftened
`np.inf` version raises.

## Why it exists: quantitative comparison against dHQP

This scenario is built to be run **head-to-head against dHQP's
`radial_switching`** on the same benchmark: both use the antipodal
start/goal layout on a circle (`build_radial_configuration` here matches
`radial_switching`'s radial crossing geometry), the same collision threshold
semantics (`d_safe`), and both are omnidirectional 2D-velocity agents.

`network_simulation.py` prints the same kind of metrics `radial_switching`
prints, in the same units, so the numbers are directly comparable:

| Printed metric | What it measures | Compare against dHQP's ... |
|---|---|---|
| `The time elapsed is ... seconds` | Total wall-clock simulation time | Same line in `radial_switching/network_simulation.py` |
| `Total solving time is ... seconds` | Sum of per-agent QP solve calls (`Agent.compute_input`) | `Total solving time is ...` (sum of dHQP's `hompc.solve_times['Solve Problem']`) |
| `Simulation stopped after N steps (T s)` | Convergence/settling time | Same convergence check style (`np.all(... < tol)`) |
| `Minimum inter-robot distance observed` | Worst-case safety margin over the whole run | The `d_safe`/formation reference lines on `distances.pdf` |

To run a fair comparison:
1. Set `n_nodes`, `dt`, `n_steps`, `v_max`, `communication_range`, and
   `d_safe` to the same values in this scenario's `settings.py` and in
   `radial_switching/settings.py`.
2. Run both `network_simulation.py` scripts and compare the printed metrics
   plus the `distances.pdf` plots (both scripts write pairwise inter-robot
   distance over time, with the safety threshold drawn as a horizontal
   line).

What the comparison is expected to show:
- **Safety**: both should keep `min distance >= d_safe` — dHQP by a genuine
  hard collision-avoidance task in its priority hierarchy, this scenario by
  the CBF soft-but-heavily-penalized constraint (which targets
  `d_safe + 2*l`, strictly more conservative than the scored `d_safe`). Since
  `w_safety` is finite, a small violation below `d_safe + 2*l` is possible
  under congestion (see "Why the safety constraint is softened, not hard"
  above) rather than a hard failure — worth checking `min_distance_m` against
  `enforced_safety_distance_m` in the comparison KPI table rather than
  assuming exact equality.
- **Task trade-off quality**: in `priority_conflict`, dHQP can make one task
  exactly dominate another (true priority); this scenario can only *bias*
  the outcome via `w_goal`/`w_form` — neither task is ever satisfied
  exactly if they conflict. Comparing final formation error / goal error
  between the two is the direct way to quantify that difference.
- **Compute cost**: dHQP's per-step solve involves the full task hierarchy
  and inter-agent consensus messaging (multiple inner-loop rounds per
  step); this scenario solves one small QP per agent per step with no
  consensus rounds, so `Total solving time` is expected to be substantially
  lower here — a legitimate lightweight-baseline selling point, at the cost
  of the soft/no-priority trade-off above.

## Key parameters (`settings.py`)

| Parameter | Meaning |
|---|---|
| `n_nodes`, `dt`, `n_steps` | Simulation size/duration |
| `communication_range` | Max sensing range for collision/formation constraints |
| `v_max` | Max commanded speed (hard box constraint + norm-clip safety net) |
| `radius` | Radius of the radial start/goal layout |
| `solver` | `QPSolver` backend (`clarabel`, `osqp`, `proxqp`, `quadprog`, `reluqp`) |
| `k_goal`, `k_form` | Gains shaping the target velocities for the goal/formation tasks |
| `d_safe`, `l`, `gamma` | Measurement threshold, extra conservative buffer (enforced = `d_safe + 2*l`), and class-K recovery gain |
| `w_safety` | Penalty weight for velocity-limit + CBF constraints (large finite, not `np.inf` -- see "Why the safety constraint is softened, not hard") |
| `d_form` | Desired inter-robot distance for a formation pair |
| `w_goal`, `w_form` | Relative weights blending the two soft equality costs (no priority order) |
| `scenario` | `'uniform'` or `'priority_conflict'` |
| `formation_pairs` | Which agent pairs must hold formation, and target distance |
| `weight_overrides` | Per-agent `(w_goal, w_form)` overrides for `priority_conflict` |
| `visual_method` | `'plot'`, `'save'`, or `'none'` |

Output visualization reuses the shared plotting/animation utilities from
`hierarchical_optimization_mpc.utils.disp_het_multi_rob`, same as the other
`distributed_ho_mpc` scenarios.
