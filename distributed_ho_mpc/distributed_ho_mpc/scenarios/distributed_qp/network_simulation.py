import copy
import os
import time
from datetime import datetime
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist

import distributed_ho_mpc.scenarios.distributed_qp.settings as st
from distributed_ho_mpc.scenarios.distributed_qp.node import Agent
from hierarchical_optimization_mpc.utils.disp_het_multi_rob import (
    MultiRobotArtistFlags,
    display_animation,
    save_snapshots,
)


def build_radial_configuration(
    n_nodes: int,
    radius: float,
    layout: str = 'symmetric',
    min_spawn_distance: float = 0.0,
    max_attempts: int = 1000,
):
    """
    Agents start on a circle and must reach the antipodal point (still on the
    circle, straight through the centre) -- goal = -start either way, since
    that point is antipodal regardless of the starting angle.

    'symmetric': evenly spaced angles (deterministic).
    'random':    random angles, resampled until every pair of agents spawns
                 at least `min_spawn_distance` apart. Used for the
                 'asymmetric' scenario. Matches
                 `orca_radial_switching.network_simulation`'s implementation
                 so the RNG draws align across methods given the same seed
                 -- same random benchmark instance for every method.
    """
    if layout == 'symmetric':
        thetas = 2 * np.pi * np.arange(n_nodes) / n_nodes
    elif layout == 'random':
        if n_nodes > 1:
            max_feasible = 2 * radius * np.sin(np.pi / n_nodes)
            if min_spawn_distance > max_feasible:
                raise ValueError(
                    f'min_spawn_distance={min_spawn_distance} is infeasible for '
                    f'{n_nodes} agents on a circle of radius={radius} '
                    f'(max possible separation is {max_feasible:.3f})'
                )
        for _ in range(max_attempts):
            thetas = np.random.uniform(0.0, 2 * np.pi, n_nodes)
            starts = [radius * np.array([np.cos(t), np.sin(t)]) for t in thetas]
            if n_nodes < 2 or pdist(np.array(starts)).min() >= min_spawn_distance:
                break
        else:
            raise RuntimeError(
                f'Could not find a random layout with min_spawn_distance='
                f'{min_spawn_distance} after {max_attempts} attempts'
            )
        goals = [-s for s in starts]
        return starts, goals
    else:
        raise ValueError(f"Unknown layout '{layout}', expected 'symmetric' or 'random'")

    starts = [radius * np.array([np.cos(t), np.sin(t)]) for t in thetas]
    goals = [-s for s in starts]
    return starts, goals


def run(out_dir: str | None = None, make_plots: bool = True) -> dict:
    """Run the weighted distributed-QP baseline and return raw trajectory/timing data.

    Kept separate from `main()` so a comparison harness can call this with
    settings monkey-patched to a shared benchmark, without going through the
    ament package-share/`out/` bookkeeping below.
    """
    np.random.seed(1)

    dt = st.dt

    fixed_starts = getattr(st, 'fixed_starts', None)
    fixed_goals = getattr(st, 'fixed_goals', None)
    if fixed_starts is not None and fixed_goals is not None:
        starts, goals = fixed_starts, fixed_goals
    else:
        layout = 'random' if st.scenario == 'asymmetric' else 'symmetric'
        starts, goals = build_radial_configuration(
            st.n_nodes,
            st.radius,
            layout=layout,
            min_spawn_distance=getattr(st, 'min_spawn_distance', 0.0),
        )

    formation_targets = {i: [] for i in range(st.n_nodes)}
    weights = {i: {'w_goal': st.w_goal, 'w_form': 0.0} for i in range(st.n_nodes)}

    if st.scenario == 'priority_conflict':
        for a, b, _ in st.formation_pairs:
            formation_targets[a].append(b)
            formation_targets[b].append(a)
        for node_id, override in st.weight_overrides.items():
            weights[node_id].update(override)

    agents = [
        Agent(
            node_id=i,
            pos=starts[i],
            goal=goals[i],
            v_max=st.v_max,
            k_goal=st.k_goal,
            d_safe=st.d_safe,
            gamma=st.gamma,
            solver=st.solver,
            formation_targets=formation_targets[i],
            k_form=st.k_form,
            d_form=st.d_form,
            w_goal=weights[i]['w_goal'],
            w_form=weights[i]['w_form'],
            d_safe_enforced=getattr(st, 'd_safe_enforced', None),
        )
        for i in range(st.n_nodes)
    ]

    num_pairs = int(st.n_nodes * (st.n_nodes - 1) / 2)
    pairwise_distances = [[] for _ in range(num_pairs)]

    s_history = [None] * st.n_steps
    last_step = st.n_steps
    goals_arr = np.array(goals)

    total_solve_time = 0.0
    min_distance = np.inf

    time_start = time.time()

    for step in range(st.n_steps):
        positions = {a.node_id: a.pos for a in agents}

        inputs = []
        for a in agents:
            t0 = time.perf_counter()
            u = a.compute_input(positions, st.communication_range)
            total_solve_time += time.perf_counter() - t0
            inputs.append(u)
        for a, u in zip(agents, inputs):
            a.step(u, dt)

        s_history[step] = [[], [copy.deepcopy(a.pos) for a in agents]]

        positions_arr = np.array([a.pos for a in agents])
        step_distances = pdist(positions_arr, metric='euclidean')
        min_distance = min(min_distance, step_distances.min())
        for i, d in enumerate(step_distances):
            pairwise_distances[i].append(d)

        goal_errors = np.linalg.norm(positions_arr - goals_arr, axis=1)
        if st.scenario == 'priority_conflict':
            # The formation pair is judged only on the formation distance,
            # not on either member's individual goal error -- one of them
            # (per `weight_overrides` above) is formation-dominant and not
            # expected to reach its own goal. Deliberately not checking
            # which one specifically: that mapping is a construction detail
            # of `weight_overrides`, not something to hardcode here.
            fa, fb, d_form = st.formation_pairs[0]
            others = [i for i in range(st.n_nodes) if i not in (fa, fb)]
            others_ok = np.all(goal_errors[others] < st.goal_tol) if others else True
            formation_dist = np.linalg.norm(positions_arr[fa] - positions_arr[fb])
            formation_ok = abs(formation_dist - d_form) < st.goal_tol
            converged_now = others_ok and formation_ok
        else:
            converged_now = np.all(goal_errors < st.goal_tol)

        if converged_now:
            last_step = step + 1
            s_history = s_history[:last_step]
            break

    time_elapsed = time.time() - time_start
    print(f'The time elapsed is {time_elapsed} seconds')
    print(f'Total solving time is {total_solve_time} seconds')
    print(f'Simulation stopped after {last_step} steps ({last_step * dt:.2f} s)')
    print(f'Minimum inter-robot distance observed: {min_distance:.4f} (d_safe = {st.d_safe})')

    if make_plots:
        assert out_dir is not None
        os.makedirs(out_dir, exist_ok=True)

        robot_pairs = list(combinations(range(st.n_nodes), 2))
        x = np.arange(1, last_step + 1) * dt
        plt.figure(figsize=(10, 6))
        for i, dist_list in enumerate(pairwise_distances):
            plt.plot(x, dist_list, label=f'Robots {robot_pairs[i]}')
        plt.axhline(y=st.d_safe, color='red', lw=2, linestyle='--', label='collision threshold')
        if st.scenario == 'priority_conflict':
            plt.axhline(y=st.d_form, color='green', lw=2, linestyle='--', label='formation target')
        plt.title(f'Pairwise robot distances -- weighted distributed QP ({st.scenario})')
        plt.xlabel('Time [s]')
        plt.ylabel('Distance [m]')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{out_dir}/distances.pdf', bbox_inches='tight', format='pdf')
        plt.close()

        flags = MultiRobotArtistFlags()
        flags.voronoi = False

        lim = st.radius + 2
        save_snapshots(
            s_history,
            goals,
            None,
            dt,
            [(last_step - 1) * dt],
            f'{out_dir}/snapshot',
            x_lim=[-lim, lim],
            y_lim=[-lim, lim],
            flags=flags,
        )

        display_animation(
            s_history,
            None,
            goals,
            None,
            dt,
            st.visual_method,
            x_lim=[-lim, lim],
            y_lim=[-lim, lim],
            video_name=f'{out_dir}/video.mp4',
            flags=flags,
        )

    return {
        'method': 'distributed_qp',
        'scenario': st.scenario,
        's_history': s_history,
        'goals': goals,
        'dt': dt,
        'last_step': last_step,
        'wall_time_s': time_elapsed,
        'solve_time_s': total_solve_time,
        'min_distance': min_distance,
        'enforced_safety_distance': agents[0].d_safe_enforced,
        'supports_priority': False,
        'supports_formation': True,
    }


def main():
    from ament_index_python.packages import get_package_share_directory

    package_name = 'distributed_ho_mpc'
    workspace_dir = f'{get_package_share_directory(package_name)}/../../../..'
    out_dir = (
        f'{workspace_dir}/out/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
        f'-distributed_qp_{st.scenario}/'
    )
    return run(out_dir=out_dir, make_plots=True)


if __name__ == '__main__':
    main()
