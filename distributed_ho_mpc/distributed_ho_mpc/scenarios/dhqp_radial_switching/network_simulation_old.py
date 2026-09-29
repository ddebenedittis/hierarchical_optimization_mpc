"""
dHQP on the aligned radial-switching benchmark.

Copied and adapted from
`radial_switching_unicycle/single_run_network_simulatio.py` (confirmed
working: its `Node.update()` correctly unpacks the 4 values
`ho_mpc_multi_robot_copy.HOMPCMultiRobot.__call__` actually returns, unlike
`radial_switching/node.py`, which is why `radial_switching_unicycle/node.py`
was copied into this folder as `node.py` instead of reusing
`radial_switching`'s). Only the internal `settings` import was changed in
that copy; the control/task logic is untouched.

Note this `Node.dual_update()` returns immediately after `save_data()` --
the numeric `rho_i` consensus update below that `return` is unreachable, and
`single_run_network_simulatio.py`'s own loop never calls
`transmit_data`/`receive_data` either. So `rho_delta` stays zero for the
whole run: each agent solves its own local task hierarchy fresh every step
from its neighbours' latest measured state, but there is no multi-round
dual/consensus exchange. That's what "confirmed working" here actually
means -- a single-shot, per-step distributed replanning scheme, not the
full iterative consensus loop `radial_switching/network_simulation.py`
attempts (and currently fails to run). The per-agent local task hierarchy
and priority reordering -- the property this comparison cares about -- is
unaffected by that simplification.
"""

import copy
import os
import time
from datetime import datetime
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist

import distributed_ho_mpc.scenarios.dhqp_radial_switching.settings as st
from distributed_ho_mpc.scenarios.dhqp_radial_switching.node import Node
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


def build_system_tasks(scenario: str, n_nodes: int, formation_pairs: list[tuple]) -> dict:
    """
    Per-agent local task hierarchy. Every agent always has input limits and
    collision avoidance at the top (safety) levels.

    Deliberately capped at 4 distinct priority levels per agent: this
    `node.py`'s solver (`ho_mpc/hierarchical_qp_copy.py:410`,
    `lamb_P = np.ones(5) * -10`) hardcodes a size-5 dual-variable array
    indexed including the priority-0 dynamics-consistency slot, so more
    than 4 actual task levels overflows it (`IndexError: index 5 is out of
    bounds for axis 0 with size 5`). The working reference example in
    `radial_switching_unicycle/single_run_network_simulatio.py` never
    exceeds 4 levels either. No `input_smooth` task here as a result (none
    of the other baselines in this comparison penalize input rate, so
    dropping it doesn't hurt comparability).

    'collision_avoidance' is fixed at prio=3, matching a second hardcoded
    constant: `Node.create_connection`/`update_task_bi` always (re)creates
    the 'collision' bi-task at `prio=3` whenever the dynamic communication
    graph changes (see `neigh_connection` below), regardless of what prio
    `Tasks()`/`MPC()` originally assigned it. Since every agent starts with
    zero neighbours here (dynamic connections), collision avoidance is
    always created via `create_connection`, never via `MPC()`'s own
    dispatch -- so prio=3 here just keeps the intended hierarchy consistent
    with what `node.py` will actually enforce, rather than silently
    documenting a value that's never used.

    'uniform': every agent's only remaining task is reaching its own goal.

    'priority_conflict': agents in a formation pair get BOTH a 'formation'
    and a 'position' (goal) task, with the priority order swapped between
    the two agents in the pair -- agent `a` values formation over its own
    goal, agent `b` values its own goal over formation. This is the direct
    dHQP analogue of the per-agent weight overrides used by the other
    baselines, expressed as an actual priority reordering instead of a gain.
    """
    tasks = {
        f'agent_{i}': [
            {'prio': 1, 'name': 'input_limits'},
            {'prio': 3, 'name': 'collision_avoidance'},
        ]
        for i in range(n_nodes)
    }

    if scenario == 'uniform':
        for i in range(n_nodes):
            tasks[f'agent_{i}'].append({'prio': 4, 'name': 'position', 'goal_index': i})
        return tasks

    if scenario != 'priority_conflict':
        raise ValueError(f"Unknown scenario '{scenario}'")

    formation_agents = set()
    for a, b, dist in formation_pairs:
        formation_agents.update((a, b))
        tasks[f'agent_{a}'].append(
            {'prio': 3, 'name': 'formation', 'agents': [[a, b]], 'distance': dist}
        )
        tasks[f'agent_{a}'].append({'prio': 4, 'name': 'position', 'goal_index': a})

        tasks[f'agent_{b}'].append({'prio': 3, 'name': 'position', 'goal_index': b})
        tasks[f'agent_{b}'].append(
            {'prio': 4, 'name': 'formation', 'agents': [[a, b]], 'distance': dist}
        )

    for i in range(n_nodes):
        if i not in formation_agents:
            tasks[f'agent_{i}'].append({'prio': 4, 'name': 'position', 'goal_index': i})

    return tasks


def neigh_connection(
    states: list[np.ndarray],
    nodes: list[Node],
    graph_matrix: np.ndarray,
    communication_range: float,
    neighbor_limit: int,
    system_tasks: dict,
) -> None:
    """
    For each node, connect to up to `neighbor_limit` nearest neighbours
    within `communication_range`; disconnect from neighbours that fall
    outside range or beyond that limit. Adapted from
    `radial_switching_unicycle/single_run_network_simulatio.py`'s
    `neigh_connection` (dropped the unused `nodes[i].a = ...` debug
    assignment; everything else is the same connect/disconnect logic
    calling `Node.create_connection`/`remove_connection`).
    """
    num_nodes = len(nodes)

    for i in range(num_nodes):
        distances = []
        for j in range(num_nodes):
            if i == j:
                continue
            dist = np.linalg.norm(states[i][:2] - states[j][:2])
            if dist < communication_range:
                distances.append((j, dist))

        distances.sort(key=lambda x: x[1])
        closest_neighbors = set(idx for idx, _ in distances[:neighbor_limit])

        current_connections = set(np.nonzero(graph_matrix[i])[0])

        to_connect = closest_neighbors - current_connections
        to_disconnect = current_connections - closest_neighbors

        for idx in to_connect:
            graph_matrix[i][idx] = 1.0
            graph_matrix[idx][i] = 1.0

            tasks_i = {f'agent_{i}': {f'agent_{idx}': copy.deepcopy(system_tasks[f'agent_{idx}'])}}
            nodes[i].create_connection(graph_matrix[i], tasks_i[f'agent_{i}'], states[idx])

            tasks_j = {f'agent_{idx}': {f'agent_{i}': copy.deepcopy(system_tasks[f'agent_{i}'])}}
            nodes[idx].create_connection(graph_matrix[idx], tasks_j[f'agent_{idx}'], states[i])

        for idx in to_disconnect:
            graph_matrix[i][idx] = 0.0
            graph_matrix[idx][i] = 0.0

            nodes[i].remove_connection(graph_matrix[i], f'agent_{idx}', idx)
            nodes[idx].remove_connection(graph_matrix[idx], f'agent_{i}', i)


def run(out_dir: str | None = None, make_plots: bool = True) -> dict:
    """Run dHQP (via the `radial_switching_unicycle`-derived `Node`) on the
    aligned radial-switching benchmark and return raw trajectory/timing
    data.

    Kept separate from `main()` so a comparison harness can call this with
    settings monkey-patched to a shared benchmark, without going through the
    ament package-share/`out/` bookkeeping below. Every agent starts with
    zero neighbours; `neigh_connection` builds/tears down the communication
    graph dynamically every step as agents move, same as
    `single_run_network_simulatio.py`.
    """
    np.random.seed(1)

    starts, goals = build_radial_configuration(st.n_nodes, st.radius)
    system_tasks = build_system_tasks(st.scenario, st.n_nodes, st.formation_pairs)

    graph_matrix = np.zeros((st.n_nodes, st.n_nodes))
    neigh_tasks = {f'agent_{i}': {} for i in range(st.n_nodes)}

    out_dir_for_nodes = out_dir or 'out'
    os.makedirs(out_dir_for_nodes, exist_ok=True)

    nodes = []
    for i in range(st.n_nodes):
        node = Node(
            i,
            graph_matrix[i],
            'omnidirectional',
            st.dt,
            system_tasks[f'agent_{i}'],
            neigh_tasks[f'agent_{i}'],
            goals,
            st.n_steps,
            out_dir=out_dir_for_nodes,
            init_s=starts[i],
        )
        nodes.append(node)
        nodes[i].Tasks()
        nodes[i].MPC()

    num_pairs = int(st.n_nodes * (st.n_nodes - 1) / 2)
    pairwise_distances = [[] for _ in range(num_pairs)]

    state = [nodes[j].s.omni[0] for j in range(st.n_nodes)]
    goals_arr = np.array(goals)
    min_distance = np.inf

    s_history = [None] * st.n_steps
    last_step = st.n_steps

    time_start = time.time()

    for step in range(st.n_steps):
        if step > 0:
            neigh_connection(
                state, nodes, graph_matrix, st.communication_range, st.neighbor_limit, system_tasks
            )
        for j in range(st.n_nodes):
            nodes[j].reorder_s_init(state)
            nodes[j].update('2')
        for j in range(st.n_nodes):
            state[j] = nodes[j].s.omni[0]
            nodes[j].dual_update()

        s_history[step] = [[], [copy.deepcopy(p) for p in state]]

        positions_arr = np.array(state)
        step_distances = pdist(positions_arr, metric='euclidean')
        min_distance = min(min_distance, step_distances.min())
        for i, d in enumerate(step_distances):
            pairwise_distances[i].append(d)

        if np.all(np.linalg.norm(positions_arr - goals_arr, axis=1) < st.goal_tol):
            last_step = step + 1
            s_history = s_history[:last_step]
            break

    time_elapsed = time.time() - time_start

    tot_solve = 0.0
    for node in nodes:
        tot_solve += node.hompc.solve_times.get('Solve Problem', 0.0)

    print(f'The time elapsed is {time_elapsed} seconds')
    print(f'Total solving time is {tot_solve} seconds')
    print(f'Simulation stopped after {last_step} steps ({last_step * st.dt:.2f} s)')

    if make_plots:
        assert out_dir is not None
        os.makedirs(out_dir, exist_ok=True)

        robot_pairs = list(combinations(range(st.n_nodes), 2))
        x = np.arange(1, last_step + 1) * st.dt
        plt.figure(figsize=(10, 6))
        for i, dist_list in enumerate(pairwise_distances):
            plt.plot(x, dist_list, label=f'Robots {robot_pairs[i]}')
        plt.axhline(y=1.0, color='red', lw=2, linestyle='--', label='collision threshold')
        if st.scenario == 'priority_conflict':
            plt.axhline(y=st.d_form, color='green', lw=2, linestyle='--', label='formation target')
        plt.title(f'Pairwise robot distances -- dHQP ({st.scenario})')
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

    return {
        'method': 'dhqp',
        'scenario': st.scenario,
        's_history': s_history,
        'goals': goals,
        'dt': st.dt,
        'last_step': last_step,
        'wall_time_s': time_elapsed,
        'solve_time_s': tot_solve,
        'min_distance': min_distance,
        'supports_priority': True,
        'supports_formation': True,
        'supports_per_agent_priority': True,
    }


def main():
    from ament_index_python.packages import get_package_share_directory

    package_name = 'distributed_ho_mpc'
    workspace_dir = f'{get_package_share_directory(package_name)}/../../../..'
    out_dir = (
        f'{workspace_dir}/out/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
        f'-dhqp_radial_switching_{st.scenario}/'
    )
    return run(out_dir=out_dir, make_plots=True)


if __name__ == '__main__':
    main()
