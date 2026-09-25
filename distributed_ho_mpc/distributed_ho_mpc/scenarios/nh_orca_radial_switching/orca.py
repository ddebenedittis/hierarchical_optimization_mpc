"""
Optimal Reciprocal Collision Avoidance (ORCA), following van den Berg et al.,
"Reciprocal n-Body Collision Avoidance" (2011).

Each agent computes, for every neighbour, a half-plane of admissible velocities
(an ORCA line) that guarantees collision-free motion for a horizon `time_horizon`
under the assumption both agents apply the reciprocal (50/50) avoidance rule.
The intersection of all half-planes (clipped to the max-speed disk) is solved
with the standard 2D linear program (successive half-plane projection, falling
back to the closest feasible point when the constraints are infeasible).
"""

import numpy as np


class HalfPlane:
    """Feasible region: points x such that det(direction, x - point) >= 0."""

    def __init__(self, point, direction):
        self.point = np.asarray(point, dtype=float)
        self.direction = np.asarray(direction, dtype=float)


def _det(a, b):
    return a[0] * b[1] - a[1] * b[0]


def _perp(v):
    return np.array([-v[1], v[0]])


def compute_orca_line(
    pos_a, vel_a, pos_b, vel_b, radius_a, radius_b, time_horizon, dt, reciprocal=True
):
    """ORCA half-plane imposed on agent A's velocity by agent B."""
    rel_pos = pos_b - pos_a
    rel_vel = vel_a - vel_b
    dist_sq = rel_pos.dot(rel_pos)
    combined_radius = radius_a + radius_b
    combined_radius_sq = combined_radius**2

    if dist_sq > combined_radius_sq:
        # Agents are not currently overlapping: build the truncated velocity-obstacle cone.
        w = rel_vel - rel_pos / time_horizon
        w_length_sq = w.dot(w)
        dot1 = w.dot(rel_pos)

        if dot1 < 0.0 and dot1**2 > combined_radius_sq * w_length_sq:
            # Closest boundary feature is the truncation circle.
            w_length = np.sqrt(w_length_sq)
            unit_w = w / w_length
            direction = np.array([unit_w[1], -unit_w[0]])
            u = (combined_radius / time_horizon - w_length) * unit_w
        else:
            # Closest boundary feature is one of the two legs of the cone.
            leg = np.sqrt(dist_sq - combined_radius_sq)
            if _det(rel_pos, w) > 0.0:
                direction = (rel_pos * leg + _perp(rel_pos) * combined_radius) / dist_sq
            else:
                direction = -(rel_pos * leg - _perp(rel_pos) * combined_radius) / dist_sq
            u = direction * rel_vel.dot(direction) - rel_vel
    else:
        # Agents already overlap: avoid collision within a single timestep.
        inv_dt = 1.0 / dt
        w = rel_vel - rel_pos * inv_dt
        w_length = np.linalg.norm(w)
        unit_w = w / w_length if w_length > 1e-9 else np.array([1.0, 0.0])
        direction = np.array([unit_w[1], -unit_w[0]])
        u = (combined_radius * inv_dt - w_length) * unit_w

    point = vel_a + (0.5 * u if reciprocal else u)
    return HalfPlane(point, direction)


def _linear_program1(lines, line_no, radius, opt_velocity, direction_opt):
    """Find the point on `lines[line_no]` (clipped to the speed disk and the
    earlier half-planes) closest to `opt_velocity`. Returns (feasible, result)."""
    dot_product = lines[line_no].point.dot(lines[line_no].direction)
    discriminant = dot_product**2 + radius**2 - lines[line_no].point.dot(lines[line_no].point)
    if discriminant < 0.0:
        return False, None

    sqrt_discriminant = np.sqrt(discriminant)
    t_left = -dot_product - sqrt_discriminant
    t_right = -dot_product + sqrt_discriminant

    for i in range(line_no):
        denominator = _det(lines[line_no].direction, lines[i].direction)
        numerator = _det(lines[i].direction, lines[line_no].point - lines[i].point)

        if abs(denominator) < 1e-9:
            if numerator < 0.0:
                return False, None
            continue

        t = numerator / denominator
        if denominator >= 0.0:
            t_right = min(t_right, t)
        else:
            t_left = max(t_left, t)
        if t_left > t_right:
            return False, None

    if direction_opt:
        if opt_velocity.dot(lines[line_no].direction) > 0.0:
            result = lines[line_no].point + t_right * lines[line_no].direction
        else:
            result = lines[line_no].point + t_left * lines[line_no].direction
    else:
        t = lines[line_no].direction.dot(opt_velocity - lines[line_no].point)
        t = min(max(t, t_left), t_right)
        result = lines[line_no].point + t * lines[line_no].direction

    return True, result


def _linear_program2(lines, radius, opt_velocity, direction_opt):
    if direction_opt:
        result = opt_velocity * radius
    else:
        norm = np.linalg.norm(opt_velocity)
        result = (
            opt_velocity * (radius / norm) if norm > radius else np.array(opt_velocity, dtype=float)
        )

    for i, line in enumerate(lines):
        if _det(line.direction, line.point - result) > 0.0:
            feasible, candidate = _linear_program1(lines, i, radius, opt_velocity, direction_opt)
            if not feasible:
                return i, result
            result = candidate

    return len(lines), result


def _linear_program3(lines, begin_line, radius, result):
    distance = 0.0
    for i in range(begin_line, len(lines)):
        if _det(lines[i].direction, lines[i].point - result) > distance:
            proj_lines = []
            for j in range(i):
                determinant = _det(lines[i].direction, lines[j].direction)
                if abs(determinant) < 1e-9:
                    if lines[i].direction.dot(lines[j].direction) > 0.0:
                        continue
                    point = 0.5 * (lines[i].point + lines[j].point)
                else:
                    t = _det(lines[j].direction, lines[i].point - lines[j].point) / determinant
                    point = lines[i].point + t * lines[i].direction

                direction = lines[j].direction - lines[i].direction
                norm = np.linalg.norm(direction)
                direction = direction / norm if norm > 1e-9 else direction
                proj_lines.append(HalfPlane(point, direction))

            perp_pref = np.array([-lines[i].direction[1], lines[i].direction[0]])
            feasible, candidate = _linear_program1(
                proj_lines, len(proj_lines) - 1, radius, perp_pref, True
            )
            if feasible:
                result = candidate
            distance = _det(lines[i].direction, lines[i].point - result)

    return result


def solve_orca_velocity(lines, max_speed, pref_velocity):
    """Solve the 2D LP: velocity closest to `pref_velocity`, inside the
    max-speed disk, satisfying every half-plane in `lines`."""
    pref_velocity = np.asarray(pref_velocity, dtype=float)
    fail_line, result = _linear_program2(lines, max_speed, pref_velocity, False)
    if fail_line < len(lines):
        result = _linear_program3(lines, fail_line, max_speed, result)
    return result


def compute_new_velocity(
    position, pref_velocity, neighbors, radius, time_horizon, max_speed, dt, velocity=None
):
    """
    Compute the ORCA-corrected velocity for one agent.

    Args:
        position: current [x, y] of the agent.
        pref_velocity: desired velocity (e.g. from the go-to-goal controller).
        neighbors: iterable of (position, velocity, radius) for every agent to avoid.
        radius: this agent's collision radius.
        time_horizon: ORCA look-ahead time tau.
        max_speed: speed limit disk radius.
        dt: simulation timestep, used only for the already-colliding branch.
        velocity: this agent's current actual velocity (defaults to pref_velocity).

    Returns:
        np.ndarray [vx, vy], the collision-free velocity closest to pref_velocity.
    """
    position = np.asarray(position, dtype=float)
    pref_velocity = np.asarray(pref_velocity, dtype=float)
    velocity = pref_velocity if velocity is None else np.asarray(velocity, dtype=float)

    lines = [
        compute_orca_line(
            position,
            velocity,
            np.asarray(n_pos, dtype=float),
            np.asarray(n_vel, dtype=float),
            radius,
            n_radius,
            time_horizon,
            dt,
        )
        for n_pos, n_vel, n_radius in neighbors
    ]

    return solve_orca_velocity(lines, max_speed, pref_velocity)
