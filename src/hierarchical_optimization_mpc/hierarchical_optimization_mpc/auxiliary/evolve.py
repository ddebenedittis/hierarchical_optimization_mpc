import numpy as np


def evolve(s: list[list[float]], u_star: list[list[float]], dt: float):
    """
    _summary_

    Args:
        s (_type_): state[robot_class_c][robot_index_j].
        u_star (_type_): optimal_input[robot_class_c][robot_index_j]
        dt (_type_): timestep

    Returns:
        s_{k+1}: evolved state
    """

    n_intervals = 10

    for c, s_c in enumerate(s):
        for j, _ in enumerate(s_c):
            if c == 0:
                for _ in range(n_intervals):
                    s[c][j] = s[c][j] + dt / n_intervals * np.array(
                        [
                            u_star[c][j][0] * np.cos(s[c][j][2]),
                            u_star[c][j][0] * np.sin(s[c][j][2]),
                            u_star[c][j][1],
                        ]
                    )

            if c == 1:
                for _ in range(n_intervals):
                    s[c][j] = s[c][j] + dt / n_intervals * np.array(
                        [
                            u_star[c][j][0],
                            u_star[c][j][1],
                        ]
                    )

    return s
