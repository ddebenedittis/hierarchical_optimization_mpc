import copy
import os
import time
from datetime import datetime
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
from ament_index_python.packages import get_package_share_directory
from scipy.spatial.distance import pdist

import distributed_ho_mpc.scenarios.distributed_qp.settings as st
from distributed_ho_mpc.scenarios.distributed_qp.node import Agent
from hierarchical_optimization_mpc.utils.disp_het_multi_rob import (
    MultiRobotArtistFlags,
    display_animation,
    save_snapshots,
)


def build_radial_configuration(n_nodes: int, radius: float):
    """Agents start on a circle and must reach the antipodal point."""
    thetas = 2 * np.pi * np.arange(n_nodes) / n_nodes
    starts = [radius * np.array([np.cos(t), np.sin(t)]) for t in thetas]
    goals = [-s for s in starts]
    return starts, goals


def main():
    np.random.seed(1)

    starts, goals = build_radial_configuration(st.n_nodes, st.radius)

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
        )
        for i in range(st.n_nodes)
    ]

    package_name = 'distributed_ho_mpc'
    workspace_dir = f'{get_package_share_directory(package_name)}/../../../..'
    out_dir = (
        f'{workspace_dir}/out/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
        f'-distributed_qp_{st.scenario}/'
    )
    os.makedirs(out_dir, exist_ok=True)

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
            a.step(u, st.dt)

        s_history[step] = [[], [copy.deepcopy(a.pos) for a in agents]]

        positions_arr = np.array([a.pos for a in agents])
        step_distances = pdist(positions_arr, metric='euclidean')
        min_distance = min(min_distance, step_distances.min())
        for i, d in enumerate(step_distances):
            pairwise_distances[i].append(d)

        if np.all(np.linalg.norm(positions_arr - goals_arr, axis=1) < st.goal_tol):
            last_step = step + 1
            s_history = s_history[:last_step]
            break

    time_elapsed = time.time() - time_start
    # Metrics printed in the same style as dHQP's radial_switching scenario,
    # for a direct quantitative comparison (wall-clock time, total QP solving
    # time, and the minimum inter-robot distance ever observed, which must
    # stay >= d_safe for the hard collision-avoidance constraint to hold).
    print(f'The time elapsed is {time_elapsed} seconds')
    print(f'Total solving time is {total_solve_time} seconds')
    print(f'Simulation stopped after {last_step} steps ({last_step * st.dt:.2f} s)')
    print(f'Minimum inter-robot distance observed: {min_distance:.4f} (d_safe = {st.d_safe})')

    robot_pairs = list(combinations(range(st.n_nodes), 2))
    x = np.arange(1, last_step + 1) * st.dt
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
        st.dt,
        [(last_step - 1) * st.dt],
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
        st.dt,
        st.visual_method,
        x_lim=[-lim, lim],
        y_lim=[-lim, lim],
        video_name=f'{out_dir}/video.mp4',
        flags=flags,
    )


if __name__ == '__main__':
    main()
