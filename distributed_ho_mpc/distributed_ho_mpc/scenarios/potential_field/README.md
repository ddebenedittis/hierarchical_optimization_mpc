# Potential Field scenario

## What this is

A minimal **artificial potential field (APF)** / behaviour-based reactive
controller, used as a **baseline to compare against dHQP** (the hierarchical
QP-based distributed controller used elsewhere in this repo, e.g.
`radial_switching`). It lives entirely in this folder:

- `node.py` — defines the `Agent` class (the control law). Despite the
  filename, this is **not** an `rclpy` ROS 2 node — it's a plain Python
  class with no ROS dependency.
- `settings.py` — scenario parameters (gains, layout, scenario variant).
- `network_simulation.py` — the runnable script (`main()`), no ROS required.

Run it directly with:
```shell
python3 src/distributed_ho_mpc/distributed_ho_mpc/scenarios/potential_field/network_simulation.py
```
There is no `console_scripts` entry point or launch file for it — it's a
standalone offline simulation, like the other scenarios under
`distributed_ho_mpc/scenarios/`.

## Is it "fully distributed"?

**The control law is distributed; the simulation harness is centralized/sequential.**

- Each `Agent.compute_input(...)` is a **pure function of that agent's own
  state** (`pos`, `goal`) plus a dict of neighbor positions **filtered by a
  communication range** — an agent ignores any neighbor farther than
  `communication_range`. There is no shared optimization problem, no central
  solver, and no coordinator deciding anything jointly across agents. In that
  sense the *control law itself* is exactly what a real distributed/decentralized
  implementation would compute on each robot.
- However, `network_simulation.py` runs everything in one Python process: it
  builds a single global `positions` dict from all agents every step and
  hands it to each `Agent.compute_input`, rather than simulating actual
  message passing / limited-bandwidth communication between agents:

  ```python
  for step in range(st.n_steps):
      positions = {a.node_id: a.pos for a in agents}
      inputs = [a.compute_input(positions, st.communication_range) for a in agents]
      for a, u in zip(agents, inputs):
          a.step(u, st.dt)
  ```

  So there's no network layer, no asynchronous updates, no packet loss/delay
  modeling — just direct dict access gated by a distance threshold to emulate
  "sensing range". This mirrors how the other scenarios in this package
  (e.g. `radial_switching`) are structured: a decentralized control law
  executed inside a centralized simulation loop for convenience.

## How the control law works

At every step, agent `i`'s commanded velocity is the **sum of independent
potential-field terms**, combined with **fixed scalar weights** (no priority
ordering between them):

```python
u = self.k_goal * (self.goal - self.pos)                      # 1. attraction

for j, p_j in positions.items():
    ...
    if dist < self.d_safe:
        u += self.k_rep * (1.0/dist - 1.0/self.d_safe) / dist**2 * (diff/dist)   # 2. repulsion

    if j in self.formation_targets:
        u += self.k_form * (dist - self.d_form) * (-diff/dist)                    # 3. formation spring

u = clip_to_max_speed(u, self.v_max)                           # 4. saturation
```

1. **Attractive term** — linear spring pulling the agent toward its goal,
   gain `k_goal`. Equivalent to the negative gradient of
   `0.5 * k_goal * ||pos - goal||^2`.
2. **Repulsive term** — classic Khatib-style APF repulsive force, active only
   once a neighbor is closer than `d_safe`:
   `k_rep * (1/dist - 1/d_safe) / dist^2 * unit_vector`, the negative gradient
   of `0.5 * k_rep * (1/dist - 1/d_safe)^2`. Neighbors farther than
   `communication_range` are ignored entirely (no sensing).
3. **Formation term** (optional) — a linear spring pulling/pushing the agent
   toward a desired inter-agent distance `d_form` from specific
   `formation_targets` neighbors. Only used in the `priority_conflict`
   scenario variant.
4. **Saturation** — the summed velocity is clipped to `v_max` by simple
   rescaling (not solved via any QP or optimization — just a norm clip).

Position integration is explicit Euler: `pos += dt * u` (`Agent.step`).

A near-zero `dist` (two agents exactly coincident) is guarded against by
jittering with a small random offset to avoid a `1/dist` singularity.

## Why it exists: comparison against dHQP

Because all terms are blended through **fixed scalar weights with no
priority structure**, this controller cannot express "always satisfy task A
before task B" the way dHQP's strict task hierarchy (cascaded QP priority
levels) does. `settings.py` has two scenario variants that make this concrete:

- **`'uniform'`** — every agent only pursues its own goal + collision
  avoidance. This is the regime potential fields are designed for, and where
  this baseline is expected to be competitive with (and cheaper than) dHQP.
- **`'priority_conflict'`** — agents 0 and 1 must also hold a formation with
  each other while still doing their radial goal-switching. Each agent gets
  an *asymmetric* weighting override (`priority_overrides`) between
  "own goal" and "hold formation" — the potential-field analogue of "give
  agent 0 and agent 1 a different priority order for the same two tasks".
  Because there's no real priority mechanism, this becomes a tug-of-war
  between fixed gains with no single weight choice that satisfies both
  agents' intent exactly — whereas dHQP handles the same setup by simply
  reordering the task hierarchy per agent, without retuning any gain.

`d_safe` and `d_form` are deliberately set to match the thresholds used in
the dHQP `radial_switching` scenario, so the two controllers can be compared
on the same benchmark geometry (`build_radial_configuration` in
`network_simulation.py` generates the same antipodal start/goal layout on a
circle, forcing trajectories to cross near the center).

## Key parameters (`settings.py`)

| Parameter | Meaning |
|---|---|
| `n_nodes`, `dt`, `n_steps` | Simulation size/duration |
| `communication_range` | Max sensing range for repulsion/formation forces |
| `v_max` | Max commanded speed (saturation) |
| `radius` | Radius of the radial start/goal layout |
| `k_goal` | Goal-attraction gain |
| `k_rep`, `d_safe` | Repulsion gain and activation distance |
| `k_form`, `d_form` | Formation spring gain and desired distance |
| `scenario` | `'uniform'` or `'priority_conflict'` |
| `formation_pairs` | Which agent pairs must hold formation, and target distance |
| `priority_overrides` | Per-agent gain overrides for the `priority_conflict` case |
| `visual_method` | `'plot'`, `'save'`, or `'none'` |

Output visualization reuses the shared plotting/animation utilities from
`hierarchical_optimization_mpc.utils.disp_het_multi_rob`, same as the other
`distributed_ho_mpc` scenarios.
