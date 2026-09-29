import copy
import itertools
import json
import os
import time
from datetime import datetime
from itertools import combinations, product
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import progressbar
from ament_index_python.packages import get_package_share_directory
from scipy.spatial.distance import pdist
from tqdm import tqdm

import distributed_ho_mpc.scenarios.radial_switching_unicycle.settings as st
from distributed_ho_mpc.scenarios.radial_switching_unicycle.node import Node
from hierarchical_optimization_mpc.utils.disp_het_multi_rob import (
    MultiRobotArtistFlags,
    display_animation,
    plot_distances,
    save_snapshots,
)
from hierarchical_optimization_mpc.utils.robot_models import (
    get_omnidirectional_model,
    get_unicycle_model,
)

DEFAULT_TASK_PRIORITIES = {
    'input_limits': 1,
    'input_smooth': 2,
    'collision_avoidance': 3,
    'position': 4,
}


def build_system_tasks(num_points, goals):
    """Build each agent's local task hierarchy.

    Under `scenario == 'uniform'` every agent gets DEFAULT_TASK_PRIORITIES. Under
    'priority_conflict' the per-agent entries in `st.priority_overrides` are
    layered on top and the pairs in `st.formation_pairs` add a shared formation
    task, so two agents can rank the same coupling task in opposite order.

    Generated from settings rather than hand-written `agent_N` literals: the
    literal form silently mismatches whenever n_nodes changes, and it duplicates
    the pairing that `formation_pairs` already states.
    """
    conflict = getattr(st, 'scenario', 'uniform') == 'priority_conflict'
    overrides = getattr(st, 'priority_overrides', {}) if conflict else {}
    pairs = getattr(st, 'formation_pairs', []) if conflict else []

    system_tasks = {}
    for ag in range(num_points):
        prios = {**DEFAULT_TASK_PRIORITIES, **overrides.get(ag, {})}
        tasks = [
            {'prio': prios['input_limits'], 'name': 'input_limits'},
            {'prio': prios['input_smooth'], 'name': 'input_smooth'},
            {'prio': prios['collision_avoidance'], 'name': 'collision_avoidance'},
            {'prio': prios['position'], 'name': 'position', 'goal': goals[ag], 'goal_index': ag},
        ]
        for a, b, distance in pairs:
            if ag in (a, b):
                tasks.append(
                    {
                        # Absent an explicit override, the formation shares the
                        # goal's level (blended, no strict order between them).
                        'prio': prios.get('formation', prios['position']),
                        'name': 'formation',
                        'agents': [[a, b]],
                        'distance': distance,
                    }
                )

        levels = sorted({t['prio'] for t in tasks})
        if len(levels) > st.n_priority:
            raise ValueError(
                f'agent_{ag} declares {len(levels)} distinct priority levels '
                f'({levels}) but n_priority={st.n_priority}; node.py sizes y_i/rho_i '
                f'by n_priority, so an extra level indexes out of bounds.'
            )
        system_tasks[f'agent_{ag}'] = tasks

    return system_tasks


def neigh_connection(
    states, nodes, graph_matrix, communication_range, limit_connection, system_tasks
):
    """
    For each node, connect to up to 6 nearest neighbors within communication range.
    Disconnect from neighbors outside range or beyond top 6 closest.
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

        # Sort and select up to 6 nearest within range
        distances.sort(key=lambda x: x[1])
        if len(distances) > 0:
            if distances[0][1] < 4:
                nodes[i].a = 1
            elif distances[0][1] > 4:
                nodes[i].a = 5
        closest_neighbors = set(idx for idx, _ in distances[:limit_connection])

        current_connections = set(np.nonzero(graph_matrix[i])[0])

        to_connect = closest_neighbors - current_connections
        to_disconnect = current_connections - closest_neighbors

        # --- CONNECT (bidirectional)
        for idx in to_connect:
            graph_matrix[i][idx] = 1.0
            graph_matrix[idx][i] = 1.0  # mirror connection

            # i connects to idx
            tasks_i = {f'agent_{i}': {f'agent_{idx}': copy.deepcopy(system_tasks[f'agent_{idx}'])}}
            nodes[i].create_connection(graph_matrix[i], tasks_i[f'agent_{i}'], states[idx])

            # idx connects to i
            tasks_j = {f'agent_{idx}': {f'agent_{i}': copy.deepcopy(system_tasks[f'agent_{i}'])}}
            nodes[idx].create_connection(graph_matrix[idx], tasks_j[f'agent_{idx}'], states[i])

        # --- DISCONNECT (bidirectional)
        for idx in to_disconnect:
            graph_matrix[i][idx] = 0.0
            graph_matrix[idx][i] = 0.0  # mirror disconnection

            # i disconnects from idx
            nodes[i].remove_connection(graph_matrix[i], f'agent_{idx}', idx)

            # idx disconnects from i
            nodes[idx].remove_connection(graph_matrix[idx], f'agent_{i}', i)


def save_run_info(output_dir: str, config: dict, run_id: str | None = None) -> Path:
    """
    Save runtime config (comm_radius, limit_conn, etc.) to a JSON file.
    Call this ONCE PER ITERATION inside your loop.

    Args:
        output_dir: The per-run output folder (e.g. root/n0_neig3_r2.5/).
        config:     Dict of parameters for this run, e.g.:
                    {"n_robot": 8, "comm_range": 2.5, "limit_conn": 3, "n": 0}
        run_id:     Optional label; auto-generated from timestamp if omitted.

    Returns:
        Path to the saved JSON file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = run_id or datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    payload = {
        'run_id': run_id,
        'timestamp': datetime.now().isoformat(),
        'config': config,
    }

    out_path = output_dir / 'run_info.json'
    out_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    # print(f"[save_run_info] Config saved → {out_path}  | params: {config}")
    return out_path


def agents_distance(state, pairwise_distances):
    """
    Plot the distance between the agents at each time step
    """
    positions_over_time = np.array(state)
    positions_over_time = positions_over_time[:, :2]
    distances = pdist(positions_over_time, metric='euclidean')  # shape: (num_pairs,)
    for i, d in enumerate(distances):
        pairwise_distances[i].append(d)
    return pairwise_distances


def count_collisions(x_hist, safety_distance=2):
    """Count distinct collision events per robot pair.

    A collision event is a maximal contiguous run of timesteps where a
    pair's distance stays below safety_distance; a single prolonged
    collision counts once, not once per timestep.

    Returns (total_events, per_pair_events) where per_pair_events maps
    (i, j) -> event count.
    """
    n_robots = x_hist.shape[1]
    per_pair_events = {}

    for i, j in itertools.combinations(range(n_robots), 2):
        dist = np.linalg.norm(x_hist[:, i] - x_hist[:, j], axis=1)
        in_collision = dist < safety_distance
        # Count rising edges (False/absent -> True).
        events = np.sum(in_collision[1:] & ~in_collision[:-1]) + int(in_collision[0])
        per_pair_events[(i, j)] = int(events)

    total_events = sum(per_pair_events.values())
    return total_events, per_pair_events


def compute_violation_stats(x_hist, safety_distance=2):
    """Compute how deep collisions cut into the safety distance.

    For every robot pair and timestep where distance < safety_distance, the
    violation is (safety_distance - distance). Returns (max_violation,
    mean_violation) across all such pair-timestep samples; both are 0.0 if
    there was never a collision.
    """
    n_robots = x_hist.shape[1]
    violations = []

    for i, j in itertools.combinations(range(n_robots), 2):
        dist = np.linalg.norm(x_hist[:, i] - x_hist[:, j], axis=1)
        in_collision = dist < safety_distance
        violations.append(safety_distance - dist[in_collision])

    violations = np.concatenate(violations) if violations else np.array([])
    if violations.size == 0:
        return 0.0, 0.0
    return float(np.max(violations)), float(np.mean(violations))


def main(
    n_robot,
    comm_range,
    limit_conn,
    root,
    cycle,
    *,
    s_init=None,
    goals=None,
    max_steps=None,
    out_dir_override=None,
    media=True,
    goal_tol=None,
):
    # np.random.seed(1)
    # b = progressbar.ProgressBar(maxval=st.n_steps, prefix=f'Simulation {cycle}: ', redirect_stdout=True)
    # b.start()

    model = {
        'unicycle': get_unicycle_model(0.5),
        'omnidirectional': get_omnidirectional_model(st.dt),
    }

    communication_range = comm_range
    limit_connection = limit_conn
    safety_distance = 2
    n_steps_eff = max_steps if max_steps is not None else st.n_steps
    # =========================================================================== #
    #                                TASK SCHEDULER                               #
    # =========================================================================== #

    # goals = [
    #         np.array([4, 6]),
    #         np.array([-6, -8]),
    #         np.array([0,0])
    #     ]
    snap = [0]  # time for snapshot
    for tt in snap:
        if tt > n_steps_eff * st.dt:
            raise ValueError('Time instant for snapshot out of simulation lenght')

    center = np.array([0, 0])  # choose the center
    num_points = n_robot

    """# radius from straight-line (chord) distance = 0.8
    radius = 5 / (2 * np.sin(np.pi / num_points))

    radius = 12
    
    goals = []
    s_init = []

    for i in range(num_points):
        theta = 2 * np.pi * i / num_points
        x = center[0] + radius * np.cos(theta)
        y = center[1] + radius * np.sin(theta)
        goals.append(np.array([x, y]))

        theta = 2 * np.pi * i / num_points - np.pi
        x = center[0] + radius * np.cos(theta)
        y = center[1] + radius * np.sin(theta)
        if st.type == 'uni':
            s_init.append(np.array([x, y, theta + np.pi]))
        if st.type == 'omni':
            s_init.append(np.array([x, y]))"""
    if s_init is None or goals is None:
        radius = 10
        min_chord = 2
        min_angle_sep = 2 * np.arcsin(min_chord / (2 * radius))  # ≈ 0.167 rad

        # Sample 8 random angles with minimum angular separation
        angles = []
        while len(angles) < 8:
            candidate = np.random.uniform(0, 2 * np.pi)
            if all(
                min(abs(candidate - a), 2 * np.pi - abs(candidate - a)) >= min_angle_sep
                for a in angles
            ):
                angles.append(candidate)

        goals = []
        s_init = []

        for theta in angles:
            x = center[0] + radius * np.cos(theta)
            y = center[1] + radius * np.sin(theta)
            goals.append(np.array([x, y]))

            theta_init = theta - np.pi
            x_init = center[0] + radius * np.cos(theta_init)
            y_init = center[1] + radius * np.sin(theta_init)

            if st.type == 'uni':
                perturbation = np.random.uniform(-np.pi / 6, np.pi / 6)  # ±45°
                s_init.append(np.array([x_init, y_init, theta_init + np.pi + perturbation]))
            if st.type == 'omni':
                s_init.append(np.array([x_init, y_init]))
    else:
        goals = [np.array(g) for g in goals]
        s_init = [np.array(s) for s in s_init]

    system_tasks = build_system_tasks(num_points, goals)

    """# system_tasks = {
    #     'agent_0': [
    #         {'prio': 1, 'name': 'input_limits'},
    #         {'prio': 2, 'name': 'input_smooth'},
    #         {'prio': 3, 'name': 'collision_avoidance'},
    #         {'prio': 4, 'name': 'position', 'goal': goals[0], 'goal_index': 0},
    #     ],
    #     'agent_1': [
    #         {'prio': 1, 'name': 'input_limits'},
    #         {'prio': 2, 'name': 'input_smooth'},
    #         {'prio': 4, 'name': 'position', 'goal': goals[1], 'goal_index': 1},
    #         {'prio': 3, 'name': 'collision_avoidance'},
    #     ],
    #     'agent_2': [
    #         {'prio': 1, 'name': 'input_limits'},
    #         {'prio': 2, 'name': 'input_smooth'},
    #         {'prio': 3, 'name': 'collision_avoidance'},
    #         {'prio': 4, 'name': 'position', 'goal': goals[2], 'goal_index': 2},
    #     ],
    #     'agent_3': [
    #         {'prio': 1, 'name': 'input_limits'},
    #         {'prio': 2, 'name': 'input_smooth'},
    #         {'prio': 3, 'name': 'collision_avoidance'},
    #         {'prio': 4, 'name': 'position', 'goal': goals[3], 'goal_index': 3},
    #     ],
    # }"""

    # ---------------------------------------------------------------------------- #
    #               Create the network and connection between agents               #
    # ---------------------------------------------------------------------------- #

    # deterministic graphs
    if st.n_nodes == 2:
        graph_matrix = np.array([[0.0, 1.0], [1.0, 0.0]])
        network_graph = nx.from_numpy_array(graph_matrix, nodelist=[0, 1])
    if st.n_nodes == 3:
        graph_matrix = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
        network_graph = nx.from_numpy_array(graph_matrix, nodelist=[0, 1, 2])
    if st.n_nodes == 4:
        graph_matrix = np.array(
            [
                [0.0, 1.0, 1.0, 1.0],
                [1.0, 0.0, 1.0, 1.0],
                [1.0, 1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0, 0.0],
            ]
        )
        network_graph = nx.from_numpy_array(graph_matrix, nodelist=[0, 1, 2, 3])
    if st.n_nodes == 5:
        graph_matrix = np.array(
            [
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
            ]
        )
        network_graph = nx.from_numpy_array(graph_matrix, nodelist=[0, 1, 2, 3, 4])
    graph_matrix = np.zeros((n_robot, n_robot))

    # random graph 🎲
    while st.random_graph:
        network_graph = nx.gnp_random_graph(st.n_nodes, st.p)
        graph_matrix = nx.to_numpy_array(network_graph)

        test = np.linalg.matrix_power((st.I_NN + graph_matrix), st.n_nodes)

        if np.all(test > 0):
            print('the graph is connected')
            # nx.draw(network_graph)
            # plt.show()
            break
        else:
            print('the graph is NOT connected')

    # update task manifold with the neighbours tasks
    neigh_tasks = {}
    for i in range(n_robot):
        id = 0
        neigh_tasks[f'agent_{i}'] = {}
        for j in graph_matrix[i]:
            if int(j) != 0:
                neigh_tasks[f'agent_{i}'][f'agent_{id}'] = copy.deepcopy(
                    system_tasks[f'agent_{id}']
                )
            id += 1

    # ----------------------------------------------------------------------------- #
    #         Create agents and initialize them based on settings and tasks        #
    # ---------------------------------------------------------------------------- #

    nodes = []  # list of agents of the system

    if out_dir_override is not None:
        out_dir = str(out_dir_override)
    else:
        package_name = 'distributed_ho_mpc'
        workspace_dir = f'{get_package_share_directory(package_name)}/../../../..'
        out_dir = f'{workspace_dir}/out/{root}{limit_conn}/{comm_range}/{cycle}/'
    os.makedirs(out_dir, exist_ok=True)
    time_start = time.time()
    # Create an agents of the same type for each node of the system
    for i in range(n_robot):
        node = Node(
            i,  # ID
            graph_matrix[i],  # Neighbours
            model['unicycle'],  # robot model
            st.dt,  # time step
            system_tasks[f'agent_{i}'],  # agent's tasks
            neigh_tasks[f'agent_{i}'],  # neighbours tasks
            goals,  # goals to be reached
            n_steps_eff,  # max simulation steps
            out_dir=out_dir,
            init_s=s_init[i],
        )
        nodes.append(node)

        # create frameworks for the agents
        nodes[i].Tasks()
        nodes[i].MPC()

    # ---------------------------------------------------------------------------- #
    #     iterate through the nodes, transmitting datas and the receiving them     #
    # ---------------------------------------------------------------------------- #

    # DISTANCES BETWEEN AGENTS
    state = [None] * n_robot  # list of x for inizialization of optimization

    num_robots = n_robot
    num_pairs = int(num_robots * (num_robots - 1) / 2)

    # Initialize one list per robot pair
    pairwise_distances = [[] for _ in range(num_pairs)]

    gg = np.array(goals[:n_robot])

    flags = MultiRobotArtistFlags()
    flags.voronoi = False
    flags.future_trajectory = False
    time_goal = 0
    start_time_coop = time.time()
    step_goal = 0
    for j in range(n_robot):
        state[j] = nodes[j].s.omni[0]  # TODO manage heterogeneous robots
    x_hist_list = [np.array(state, dtype=float)]
    u_hist_list = []
    solve_times_arr = np.zeros((n_steps_eff, n_robot))
    for i in tqdm(range(n_steps_eff), desc='iteration', position=2, colour='red', leave=False):
        #    for i in range(n_steps_eff):
        if i == n_steps_eff - 1:
            last_step = i + 1
        if i > 0:
            neigh_connection(
                state, nodes, graph_matrix, communication_range, limit_connection, system_tasks
            )
        # for rr in range(st.inner_loop):
        for j in range(n_robot):
            nodes[j].reorder_s_init(state)
            _t0 = time.perf_counter()
            nodes[j].update('2')  # Update primal solution and state evolution
            solve_times_arr[i, j] = time.perf_counter() - _t0
        u_hist_list.append(np.array([nodes[j].u_star[0][0] for j in range(n_robot)], dtype=float))
        for j in range(n_robot):
            state[j] = nodes[j].s.omni[0]  # TODO manage heterogeneous robots
            # for ij in nodes[j].neigh:  # select my neighbours
            #     msg = nodes[j].transmit_data(ij, 'P')  # Transmit primal variable
            #     nodes[ij].receive_data(msg)  # neighbour receives the message
            # for j in range(st.n_nodes):
            nodes[j].dual_update()  # linear update of dual problem
        # for j in range(st.n_nodes):
        #     for ij in nodes[j].neigh:  # select my neighbours
        #         msg = nodes[j].transmit_data(ij, 'D')  # Transmit Dual variable
        #         nodes[ij].receive_data(msg)  # neighbour receives the message
        # for j in range(st.n_nodes):
        #     nodes[j].reorder_s_init(state)
        #     nodes[j].update('2')  # Update primal solution and state evolution
        # if i%100 == 0 and i > 0:
        #     s_hist_merged = [
        #         sum(([node.s_history[i][0][0]] for node in nodes), [])
        #         for ll in range(len(nodes[0].s_history))
        #     ]
        #     s_hist_merged = [[[], s_k] for s_k in s_hist_merged]
        #     save_snapshots(
        #         s_hist_merged,
        #         goals,
        #         None,
        #         st.dt,
        #         [(i - 1) * st.dt],
        #         f'{out_dir}/snapshot_{i}',
        #         x_lim=[-10, 10],
        #         y_lim=[-10, 10],
        #         flags=flags,
        #     )
        x_hist_list.append(np.array(state, dtype=float))
        if goal_tol is None:
            goal_reached = np.all(np.abs(np.array(state)[:, :2] - gg) < 2e-2)
        else:
            goal_reached = np.all(np.linalg.norm(np.array(state)[:, :2] - gg, axis=1) <= goal_tol)
        if goal_reached and step_goal == 0:
            last_step = i + 1
            for j in range(n_robot):
                nodes[j].s_history = nodes[j].s_history[:last_step]
            time_goal = time.time() - time_start
            step_goal = i
            break
        # b.update(i)

    time_elapsed = time.time() - time_start
    if time_goal == 0:
        time_goal = time_elapsed
        step_goal = n_steps_eff
    time_coop = time.time() - start_time_coop
    # print(f'The time elapsed is {time_elapsed} seconds')
    # print(f'Time used to coordinate the network is {time_coop}')
    # print('The time was used in the following phases:')
    tot_creation = 0
    tot_solve = 0
    max_value = []
    creation = []
    for n, agent in enumerate(nodes):
        max_key_len = max(map(len, agent.hompc.solve_times.keys()))
        for key, value in agent.hompc.solve_times.items():
            key_len = len(key)
            if key == 'Create Problem':
                tot_creation += value
                creation.append(value)
            if key == 'Solve Problem':
                tot_solve += value
                max_value.append(value)
            # print(f"agent{n} {key}: {' '*(max_key_len-key_len)}{value}")
    # print(f'Total creation time is {tot_creation}s')
    # print(f'Total solving time is {tot_solve}s')
    max_a = max(max_value)

    # with open(f'{out_dir}time.txt', 'w') as file:
    #     file.write(
    #         f'dt: {st.dt}\n n_c: {st.n_control}\ntime elapsed: {time_elapsed}s\ntotal solving {tot_solve}\n time to goal: {time_goal} at iter {step_goal} \nmax {max_a}\nall max {max_value}\n cr {creation}'
    #     )

    if media and st.simulation:
        """robot_pairs = list(combinations(range(num_robots), 2))
        x = np.arange(1, last_step + 1) * st.dt
        plt.figure(figsize=(10, 6))
        for i, dist_list in enumerate(pairwise_distances):
            plt.plot(x, dist_list, label=f'Robots {robot_pairs[i]}')
        plt.axhline(y=2, color='green', lw=3, linestyle='--')
        plt.title('Time Evolution of Pairwise Robot Distances')
        plt.xlabel('Time Step')
        plt.ylabel('Distance')
        # plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{out_dir}/distances.pdf', bbox_inches='tight', format='pdf')
        plt.close()"""

        # ---------------------------------------------------------------------------- #
        #                          plot the states evolutions                          #
        # ---------------------------------------------------------------------------- #
        s_hist_merged = [
            sum(([node.s_history[i][0][0]] for node in nodes), [])
            for i in range(len(nodes[0].s_history))
        ]
        if st.type == 'omni':
            s_hist_merged = [[[], s_k] for s_k in s_hist_merged]
        elif st.type == 'uni':
            s_hist_merged = [[s_k, []] for s_k in s_hist_merged]

        s_history_all = []
        for r in range(len(nodes[0].state_k)):
            s = []
            for i in range(num_robots):
                s_k = []
                for k in range(1, st.n_control):
                    s_k.append(nodes[i].state_k[r][k])
                s.append(s_k)
            if st.type == 'omni':
                s_history_all.append([[], s])
            elif st.type == 'uni':
                s_history_all.append([s, []])

        # n_k = len(s_history_all)
        # n_c = len(s_history_all[0])
        # n_j = [len(s_history_all[0][c]) for c in range(n_c)]
        # n_coord = 2
        # x_hist = np.zeros([n_k, sum(n_j), n_coord])

        # for k, c in np.ndindex(n_k, n_c):
        #     for j in range(n_j[c]):
        #         x_hist[k, sum(n_j[:c]) + j] = s_history_all[k][c][j][:n_coord]
        # for i, j in itertools.combinations(range(sum(n_j)), 2):
        #     pairwise_distances.append(np.maximum(np.linalg.norm(x_hist[:, i] - x_hist[:, j], axis=1), 0.1))

        # total_events, _ = count_collisions(x_hist, safety_distance)
        # max_violation, mean_violation = compute_violation_stats(x_hist, safety_distance)

        """distances = [[] for n in range(st.n_nodes)]
        for iter in s_hist_merged:
            for nn, ag in enumerate(iter[0]):
                dist_opt = np.linalg.norm(ag[:2] - centr_sol[nn])
                distances[nn].append(dist_opt)

        # distances = np.array(distances)  # shape: (n_valid_iterations, 4)

        plt.figure(figsize=(8, 5))
        for nn, fig in enumerate(distances):
            plt.semilogy(fig, label=f'agent{nn}')
        plt.xlabel('Iteration')
        plt.ylabel('Distance (log scale)')
        plt.title('Distances per vector')
        # plt.legend()
        plt.grid(True, which='both', ls='--')
        plt.savefig(f'{out_dir}/dist_to_opt.pdf', bbox_inches='tight', format='pdf')
        plt.close()"""

        flags = MultiRobotArtistFlags()
        flags.future_trajectory = False
        flags.voronoi = False
        flags.legend = False
        flags.centroid = False

        save_snapshots(
            s_hist_merged,
            goals,
            None,
            st.dt,
            [(last_step - 1) * st.dt],
            f'{out_dir}/snapshot',
            x_lim=[-11, 11],
            y_lim=[-11, 11],
            flags=flags,
        )
        # display_animation(
        #     s_hist_merged,
        #     s_history_all,
        #     goals,
        #     None,
        #     st.dt,
        #     st.visual_method,
        #     video_name=f'{out_dir}/video.mp4',
        #     x_lim=[-11, 11],
        #     y_lim=[-11, 11],
        #     flags=flags,
        # )
        # plot_distances(
        #     s_hist_merged,
        #     0.05,  # dt
        #     2,  # dmin
        #     f'{out_dir}/distances.pdf',
        #     to_obj=False,
        #     form=False,
        # )

    # Unique subfolder per iteration
    run_label = f'n{cycle}_neig{limit_conn}_r{comm_range}'

    # ── Save config for THIS iteration ────────────────────────
    save_run_info(
        output_dir=out_dir,
        config={
            'n_robot': n_robot,
            'comm_range': comm_range,
            'limit_conn': limit_conn,
            'id': cycle,
            'dt': st.dt,
            'max_steps': n_steps_eff,
            'n_control': st.n_control,
            'time_elapsed': time_elapsed,
            'total_solve': tot_solve,
            'time_to_goal': time_goal,
            'iter_to_goal': step_goal,
            'max_solve_time': max_a,
            'safety_distance': 2,
            # "total_collisions": total_events,
            # "max_violation": max_violation,
            # "mean_violation": mean_violation,
            # "all_max_solve_times": max_value,
            # "creation_times": creation,
        },
        run_id=run_label,
    )

    # b.finish()

    return {
        'x_hist': np.array(x_hist_list),
        'u_hist': np.array(u_hist_list),
        'solve_time_total': tot_solve,
        'solve_time_max': max(agent.hompc.max_iter for agent in nodes),
        'n_solves': len(u_hist_list) * n_robot,
        'steps': len(u_hist_list),
        'solve_times': solve_times_arr[: len(u_hist_list)],
        'creation_time_total': tot_creation,
    }


if __name__ == '__main__':
    package_name = 'distributed_ho_mpc'
    workspace_dir = f'{get_package_share_directory(package_name)}/../../../..'
    date_str = f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}-radial_switching/'
    root = f'{workspace_dir}/out/{date_str}'
    os.makedirs(root, exist_ok=True)
    n_robot = 8
    n_neig = [1, 2, 3, 4, 5, 6, 7]
    comm_radius = [2.5, 5, 10]
    cycles = 51

    for nn in tqdm(range(20, cycles), desc='Cycles', position=0, colour='blue'):
        for ii, jj in tqdm(
            product(n_neig, comm_radius),
            desc='  Simulation',
            position=1,
            leave=False,
            total=len(n_neig) * len(comm_radius),
            colour='green',
        ):
            main(n_robot, jj, ii, date_str, nn)

    print('Im tired bro, simulation done. going to sleep')
