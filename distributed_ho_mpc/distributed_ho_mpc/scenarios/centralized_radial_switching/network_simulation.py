import copy
import os
import time
from datetime import datetime
from itertools import combinations

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist

import distributed_ho_mpc.scenarios.centralized_radial_switching.settings as st
from hierarchical_optimization_mpc.ho_mpc_multi_robot import (
    HOMPCMultiRobot,
    TaskBiCoeff,
    TaskType,
)
from hierarchical_optimization_mpc.utils.disp_het_multi_rob import (
    MultiRobotArtistFlags,
    display_animation,
    save_snapshots,
)
from hierarchical_optimization_mpc.utils.robot_models import RobCont, get_omnidirectional_model


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


def evolve(s: RobCont, u_star: RobCont, dt: float, n_intervals: int = 10) -> RobCont:
    for j in range(len(s.omni)):
        for _ in range(n_intervals):
            s.omni[j] = s.omni[j] + dt / n_intervals * np.array(
                [u_star.omni[j][0], u_star.omni[j][1]]
            )
    return s


def run(out_dir: str | None = None, make_plots: bool = True) -> dict:
    """Run the centralized-HQP baseline on the aligned radial-switching
    benchmark and return raw trajectory/timing data.

    Kept separate from `main()` so a comparison harness can call this with
    settings monkey-patched to a shared benchmark, without going through the
    ament package-share/`out/` bookkeeping below.
    """
    np.random.seed(1)

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
    n_robots = st.n_nodes

    # Symbolic state/input/dynamics used to build the HOMPC problem.
    s = RobCont(omni=None)
    u = RobCont(omni=None)
    s_kp1 = RobCont(omni=None)
    s.omni, u.omni, s_kp1.omni = get_omnidirectional_model(st.dt)

    hompc = HOMPCMultiRobot(s.tolist(), u.tolist(), s_kp1.tolist(), RobCont(omni=n_robots).tolist())
    hompc.n_control = st.n_control
    hompc.n_pred = st.n_pred

    task_input_limits = RobCont(
        omni=ca.vertcat(
            u.omni[0] - st.v_max,
            -u.omni[0] - st.v_max,
            u.omni[1] - st.v_max,
            -u.omni[1] - st.v_max,
        )
    )
    hompc.create_task(
        name='input_limits',
        prio=1,
        type=TaskType.Same,
        ineq_task_ls=task_input_limits.tolist(),
    )

    aux_avoid_collision = ca.SX.sym('aux', 2, 2)
    mapping_avoid_collision = RobCont(omni=ca.vertcat(s.omni[0], s.omni[1]))
    task_avoid_collision = ca.vertcat(
        -((aux_avoid_collision[0, 0] - aux_avoid_collision[1, 0]) ** 2)
        - (aux_avoid_collision[0, 1] - aux_avoid_collision[1, 1]) ** 2,
    )
    task_avoid_collision_coeff = [
        TaskBiCoeff(0, i, 0, j, 0, -(st.d_safe**2))
        for i in range(n_robots)
        for j in range(i + 1, n_robots)
    ]
    hompc.create_task_bi(
        name='collision_avoidance',
        prio=2,
        type=TaskType.Bi,
        aux=aux_avoid_collision,
        mapping=mapping_avoid_collision.tolist(),
        ineq_task_ls=task_avoid_collision,
        ineq_task_coeff=task_avoid_collision_coeff,
    )

    position_prio = 3
    formation_prio = 3

    if st.scenario == 'priority_conflict':
        # A centralized hierarchy is global: there is no per-agent priority
        # order to set. Formation is prioritized above every individual
        # goal fleet-wide -- the only unambiguous analogue of dHQP's
        # per-agent conflict a single shared hierarchy can express.
        formation_prio = 3
        position_prio = 4

        aux_form = ca.SX.sym('aux_form', 2, 2)
        mapping_form = RobCont(omni=ca.vertcat(s.omni[0], s.omni[1]))
        task_formation = ca.vertcat(
            (aux_form[0, 0] - aux_form[1, 0]) ** 2 + (aux_form[0, 1] - aux_form[1, 1]) ** 2,
        )
        for a, b, dist in st.formation_pairs:
            hompc.create_task_bi(
                name=f'formation_{a}_{b}',
                prio=formation_prio,
                type=TaskType.Bi,
                aux=aux_form,
                mapping=mapping_form.tolist(),
                eq_task_ls=task_formation,
                eq_task_coeff=[TaskBiCoeff(0, a, 0, b, 0, dist**2)],
            )

    for i in range(n_robots):
        task_pos = RobCont(omni=ca.vertcat(s_kp1.omni[0], s_kp1.omni[1]))
        task_pos_coeff = RobCont(omni=[[goals[i]] for _ in range(n_robots)])
        hompc.create_task(
            name=f'position_{i}',
            prio=position_prio,
            type=TaskType.Same,
            eq_task_ls=task_pos.tolist(),
            eq_task_coeff=task_pos_coeff.tolist(),
            robot_index=[[i]],
        )

    num_pairs = int(n_robots * (n_robots - 1) / 2)
    pairwise_distances = [[] for _ in range(num_pairs)]

    s_history = [None] * st.n_steps
    last_step = st.n_steps
    goals_arr = np.array(goals)
    min_distance = np.inf

    # Numeric state used to drive the simulation loop, separate from the
    # symbolic `s` used above to build the HOMPC problem.
    state = RobCont(omni=[np.array(p) for p in starts])

    time_start = time.time()

    for step in range(st.n_steps):
        u_0, _ = hompc(copy.deepcopy(state.tolist()))
        state = evolve(state, RobCont(omni=u_0[0]), st.dt)

        s_history[step] = [[], [copy.deepcopy(p) for p in state.omni]]

        positions_arr = np.array(state.omni)
        step_distances = pdist(positions_arr, metric='euclidean')
        min_distance = min(min_distance, step_distances.min())
        for i, d in enumerate(step_distances):
            pairwise_distances[i].append(d)

        goal_errors = np.linalg.norm(positions_arr - goals_arr, axis=1)
        if st.scenario == 'priority_conflict':
            # A centralized hierarchy is global, not per-agent (see the
            # 'priority_conflict' comment in settings.py): formation is
            # prioritized above BOTH agents' individual goals, so neither
            # `fa` nor `fb` is individually required to reach its own goal
            # here -- only the formation constraint and everyone else.
            fa, fb, d_form = st.formation_pairs[0]
            others = [i for i in range(n_robots) if i not in (fa, fb)]
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
    solve_time = hompc.solve_times.get('Solve Problem', None)
    print(f'The time elapsed is {time_elapsed} seconds')
    print(f'Total solving time is {solve_time} seconds')
    print(f'Simulation stopped after {last_step} steps ({last_step * st.dt:.2f} s)')

    if make_plots:
        assert out_dir is not None
        os.makedirs(out_dir, exist_ok=True)

        robot_pairs = list(combinations(range(n_robots), 2))
        x = np.arange(1, last_step + 1) * st.dt
        plt.figure(figsize=(10, 6))
        for i, dist_list in enumerate(pairwise_distances):
            plt.plot(x, dist_list, label=f'Robots {robot_pairs[i]}')
        plt.axhline(y=st.d_safe, color='red', lw=2, linestyle='--', label='collision threshold')
        if st.scenario == 'priority_conflict':
            plt.axhline(y=st.d_form, color='green', lw=2, linestyle='--', label='formation target')
        plt.title(f'Pairwise robot distances -- centralized HQP ({st.scenario})')
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
        'method': 'centralized_hqp',
        'scenario': st.scenario,
        's_history': s_history,
        'goals': goals,
        'dt': st.dt,
        'last_step': last_step,
        'wall_time_s': time_elapsed,
        'solve_time_s': solve_time,
        'min_distance': min_distance,
        'supports_priority': True,
        'supports_formation': True,
        'supports_per_agent_priority': False,
    }


def main():
    from ament_index_python.packages import get_package_share_directory

    package_name = 'distributed_ho_mpc'
    workspace_dir = f'{get_package_share_directory(package_name)}/../../../..'
    out_dir = (
        f'{workspace_dir}/out/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
        f'-centralized_radial_switching_{st.scenario}/'
    )
    return run(out_dir=out_dir, make_plots=True)


if __name__ == '__main__':
    main()
