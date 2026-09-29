# NH-ORCA Radial Switching scenario

## What this is

A **distributed, per-agent ORCA (Optimal Reciprocal Collision Avoidance)
controller** with an explicit `epsilon` conservatism margin on top of the
plain reciprocity radius (see "Margin: why NH-ORCA enforces more than it's
measured against" below), run on the same radial-switching benchmark as
[`potential_field`](../potential_field/README.md) and
[`cbf_qp`](../cbf_qp/README.md): agents start on a circle and
must reach the antipodal point, forcing every pair of trajectories to cross
near the centre. The start layout can be the classic **evenly-spaced**
configuration or a **randomized** one (random angle per agent, goal still the
antipodal point, spawn points kept at least `min_spawn_distance` apart) to
break the rotational symmetry that the even spacing hides. It plays the same
"lightweight
baseline" role as those two scenarios, but replaces their heuristic
repulsion/QP safety filter with the actual reciprocal-velocity-obstacle
algorithm (van den Berg et al., 2011) used in crowd/multi-robot navigation.
It lives entirely in this folder:

- `orca.py` — self-contained ORCA math: half-plane (velocity obstacle)
  construction per neighbour pair, and the 2D linear program that finds the
  velocity closest to the preferred one satisfying every half-plane plus the
  max-speed disk. No ROS/casadi/qpsolvers dependency.
- `node.py` — defines the `Agent` class (the control law). Not an `rclpy`
  ROS 2 node — a plain Python class with no ROS dependency.
- `settings.py` — scenario parameters (layout, go-to-goal gain, ORCA
  parameters, visualization).
- `network_simulation.py` — the runnable script (`main()`), no ROS required;
  reuses `hierarchical_optimization_mpc.utils.disp_het_multi_rob` for the
  distance plot, snapshot, and animation, same as every other scenario here.

Run it directly with:
```shell
python3 src/distributed_ho_mpc/distributed_ho_mpc/scenarios/nh_orca_radial_switching/network_simulation.py
```
There is no `console_scripts` entry point or launch file for it — it's a
standalone offline simulation, like the other scenarios under
`distributed_ho_mpc/scenarios/`.

## Is it "fully distributed"?

**The control law is distributed; the simulation harness is
centralized/sequential** — the same split as `potential_field` and
`cbf_qp`.

- Each `Agent.compute_input(...)` uses only its own state (`pos`, `goal`,
  `velocity`) and a dict of neighbour positions/velocities **filtered by
  `communication_range`**. There is no shared/joint optimization problem, no
  central solver, and no dual-variable/consensus exchange between agents
  (unlike `radial_switching`'s dHQP).
- `network_simulation.py` still builds a single global `positions` /
  `velocities` dict from all agents every step and hands it to each
  `Agent.compute_input`, rather than simulating actual message passing
  between agents — a decentralized control law executed inside a
  centralized simulation loop for convenience, exactly like the other
  scenarios in this package.

## How the control law works

At every step, agent `i`:

1. Computes a **preferred velocity** — a simple proportional go-to-goal law,
   saturated to `v_max`:
   ```python
   pref = clip(k_goal * (goal - pos), max_norm=v_max)
   ```
2. Builds one **ORCA half-plane per sensed neighbour** (`orca.py`), derived
   from the reciprocal (50/50) velocity obstacle between the two agents —
   using the neighbour's *last committed* velocity, since its preferred
   velocity for this step isn't known to us (this mirrors what a real
   distributed ORCA agent would sense/broadcast).
3. Solves the **2D linear program** for the velocity closest to `pref` that
   satisfies every half-plane and the `orca_max_speed` disk — this is the
   agent's actual commanded velocity.
4. Integrates position with explicit Euler: `pos += dt * u` (`Agent.step`).

Unlike `potential_field`'s summed repulsive/attractive forces or
`cbf_qp`'s single per-agent QP safety filter, ORCA's collision
avoidance is a genuine **multi-agent reciprocity guarantee**: as long as
both agents in a pair run ORCA and share the same `orca_radius`, neither
agent needs to fully react to a closing agent — geometrically, `2 * orca_radius`
separation is preserved without either side over- or under-braking. This was
verified standalone (two head-on agents deflect and settle at exactly
`2 * orca_radius` apart) and on the full 5-agent radial-switching benchmark
(minimum pairwise distance across the run equals `2 * orca_radius`, i.e. no
collision and no unnecessary slack).

## Margin: why NH-ORCA enforces more than it's measured against

`d_safe` is the shared measurement threshold every method in `comparison/`
is scored against. `epsilon` is an extra conservative buffer (absorbing
tracking error) this baseline holds internally, so `orca_radius` is derived
as `(d_safe + 2*epsilon) / 2.0` instead of exactly `d_safe / 2.0` — the
enforced pairwise collision diameter (`2 * orca_radius`) ends up strictly
larger than `d_safe`. This is the "enforced > measured" asymmetry a
real NH-ORCA controller needs to stay safe under nonzero tracking error,
reproduced here as an explicit constant rather than left implicit.

Honest caveat: the agents in this scenario remain **holonomic
single-integrators** (`u = [vx, vy]`, no heading state) — there is no
nonholonomic trackability constraint (heading-tracking cone/disc) modeled
here, unlike the literature NH-ORCA extension for unicycle/differential-drive
robots. "NH" in this scenario's name carries over the margin convention from
that literature, not a kinematic constraint; if unicycle kinematics are ever
added to this benchmark, the heading-tracking restriction would need to be
implemented separately.

## Key parameters (`settings.py`)

| Parameter | Meaning |
|---|---|
| `n_nodes`, `dt`, `n_steps` | Simulation size/duration |
| `communication_range` | Max sensing range — neighbours farther than this are ignored |
| `v_max` | Max commanded speed (preferred-velocity saturation) |
| `radius` | Radius of the radial start/goal layout |
| `layout` | `'symmetric'` (evenly spaced) or `'random'` (random angle per agent, resampled to respect `min_spawn_distance`) |
| `min_spawn_distance` | Minimum spawn separation enforced when `layout == 'random'` (raises if infeasible for the given `n_nodes`/`radius`) |
| `k_goal` | Go-to-goal proportional gain |
| `goal_tol` | Stop tolerance (distance to goal) |
| `d_safe`, `epsilon` | Measurement threshold and extra conservative buffer (`orca_radius = (d_safe + 2*epsilon) / 2`) |
| `orca_radius` | Collision radius used to build each agent's velocity obstacle (derived from `d_safe`/`epsilon` above) |
| `orca_time_horizon` | Look-ahead time `tau` — smaller reacts later/more sharply, larger avoids earlier/more conservatively |
| `orca_max_speed` | Speed-disk radius used by the ORCA LP (defaults to `v_max`) |
| `visual_method` | `'plot'`, `'save'`, or `'none'` |

Output visualization reuses the shared plotting/animation utilities from
`hierarchical_optimization_mpc.utils.disp_het_multi_rob`, same as the other
`distributed_ho_mpc` scenarios.
