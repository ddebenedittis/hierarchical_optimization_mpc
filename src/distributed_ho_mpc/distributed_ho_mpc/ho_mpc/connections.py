"""Communication-graph maintenance shared by every dHQP scenario.

The link rule is the mutual (AND) k-nearest rule: robots i and j are linked iff
each is among the other's `limit_connection` nearest robots within
`communication_range`. It is symmetric by construction, so both endpoints agree
on every link, and it bounds every robot's degree by `limit_connection`, which
bounds the size of its local QP. The price is that a robot can end up with fewer
than `limit_connection` links, e.g. when a close robot ranks it outside its own
top-k; that relaxes the global problem (a collision pair may be omitted) but
never tightens it.
"""

import copy
from itertools import combinations

import numpy as np

from distributed_ho_mpc.ho_mpc.ho_mpc_multi_robot_copy import TaskType


def mutual_knn_links(
    positions: np.ndarray, communication_range: float, limit_connection: int
) -> np.ndarray:
    """Symmetric boolean adjacency of the mutual k-nearest graph within range.

    Args:
        positions: (n, 2) robot positions.
        communication_range: links only form between robots closer than this.
        limit_connection: maximum number of links per robot.

    Returns:
        (n, n) boolean adjacency with a false diagonal and max row sum
        `limit_connection`.
    """
    n = len(positions)
    dist = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=2)

    top_k = np.zeros((n, n), dtype=bool)
    for i in range(n):
        in_range = [
            j
            for j in np.argsort(dist[i], kind='stable')
            if j != i and dist[i, j] < communication_range
        ]
        top_k[i, in_range[:limit_connection]] = True

    return top_k & top_k.T


def update_links(
    states,
    nodes,
    graph_matrix: np.ndarray,
    communication_range: float,
    limit_connection: int,
    system_tasks: dict,
) -> tuple[int, int]:
    """Bring `graph_matrix` and the nodes' local problems to the mutual k-nearest graph.

    The target link set is computed once from the current positions and only the
    difference to the current graph is applied, disconnections first, so links
    that are still wanted are never torn down and rebuilt. Call it at every step,
    including step 0: the graph starts empty.

    Returns:
        (n_connect, n_disconnect) link events applied by this call.
    """
    positions = np.array([np.asarray(s, dtype=float)[:2] for s in states])
    target = mutual_knn_links(positions, communication_range, limit_connection)
    current = graph_matrix != 0

    pairs = list(combinations(range(len(nodes)), 2))
    to_disconnect = [(i, j) for i, j in pairs if current[i, j] and not target[i, j]]
    to_connect = [(i, j) for i, j in pairs if target[i, j] and not current[i, j]]

    for i, j in to_disconnect:
        graph_matrix[i][j] = 0.0
        graph_matrix[j][i] = 0.0
        nodes[i].remove_connection(graph_matrix[i], f'agent_{j}', j)
        nodes[j].remove_connection(graph_matrix[j], f'agent_{i}', i)

    for i, j in to_connect:
        graph_matrix[i][j] = 1.0
        graph_matrix[j][i] = 1.0
        nodes[i].create_connection(
            graph_matrix[i], {f'agent_{j}': copy.deepcopy(system_tasks[f'agent_{j}'])}, states[j]
        )
        nodes[j].create_connection(
            graph_matrix[j], {f'agent_{i}': copy.deepcopy(system_tasks[f'agent_{i}'])}, states[i]
        )

    return len(to_connect), len(to_disconnect)


# ============================ Node-side helpers ============================= #


def own_prio(tasks: list[dict], name: str, default: int | None = None) -> int:
    """Priority of task `name` in an agent's own task list, or `default` if it has none."""
    prio = next((t['prio'] for t in tasks if t['name'] == name), default)
    if prio is None:
        raise KeyError(f'task {name!r} is not in the agent task list and has no default')
    return prio


def has_formation_task(hompc, prio: int, robot_index: list[list[int]]) -> bool:
    """Whether `hompc` already holds a formation task with this priority and robot pair."""
    return any(
        t.name == 'formation' and t.prio == prio and t.robot_index == robot_index
        for t in hompc._tasks
    )


def keep_on_removal(task, id_to_remove: int, degree: int) -> bool:
    """Whether a task survives the removal of local robot `id_to_remove`.

    Dropped: tasks that exist only because of that robot, i.e. a pairwise task
    involving it (formation) or a task on it alone (its goal copy, its vel_ref).
    Kept: tasks spanning all local robots (input limits, input smoothing,
    obstacle avoidance), which are re-indexed afterwards, and the collision task,
    which is rebuilt, unless no neighbour is left.
    """
    if task.name == 'collision':
        return degree > 0
    robots = task.robot_index[0]
    if task.type == TaskType.Bi:
        return id_to_remove not in robots
    return robots != [id_to_remove]


def remap_robot_index(
    robots: list[int], robot_idx_global_old: list[int], robot_idx_global: list[int]
):
    """Re-express local robot indices after a removal, dropping the removed robot."""
    return [
        robot_idx_global.index(robot_idx_global_old[r])
        for r in robots
        if robot_idx_global_old[r] in robot_idx_global
    ]
