"""NH-ORCA baseline: nonholonomic Optimal Reciprocal Collision Avoidance.

Reactive, decentralized collision avoidance following Alonso-Mora et al.,
"Optimal Reciprocal Collision Avoidance for Multiple Non-Holonomic Robots"
(DARS 2010) built on top of the ORCA half-plane construction from van den
Berg et al. / the RVO2 library. Each robot solves a small 2-D QP for a
holonomic velocity that respects reciprocal collision-avoidance half-planes,
then tracks that velocity with a unicycle controller.
"""

from __future__ import annotations

import math

import numpy as np
import qpsolvers

from distributed_ho_mpc.scenarios.comparison.common.base_agent import BaseAgent, run_reactive
from distributed_ho_mpc.scenarios.comparison.common.benchmark import BenchmarkInstance
from distributed_ho_mpc.scenarios.comparison.common.run_io import RunResult


def _wrap_to_pi(angle: float) -> float:
    """Wrap an angle to the interval [-pi, pi)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _orca_u(
    rel_pos: np.ndarray, rel_vel: np.ndarray, r: float, tau: float, dt: float
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the RVO2 ORCA correction vector and half-plane normal for one neighbor.

    Case analysis (mirrors RVO2's `Agent::computeNewVelocity`):

    1. Non-colliding (``||rel_pos|| > r``): the velocity obstacle is a
       truncated cone -- a disc of radius ``r/tau`` centered at
       ``rel_pos/tau`` (the "cutoff circle"), capped by two tangent legs
       from the origin to that disc.
       - If the relative velocity's nearest way out of the VO passes
         through the cutoff disc (checked via the sign/magnitude of the
         projection ``w . rel_pos``), project onto the disc boundary.
       - Otherwise project onto the nearer of the two legs, selected by
         the sign of ``det(rel_pos, w)`` (left leg vs. right leg).
    2. Colliding (``||rel_pos|| <= r``): the robots already overlap, so
       fall back to a disc of radius ``r/dt`` centered at ``rel_pos/dt``
       (a one-time-step cutoff) so the pair can still separate.

    In both cases ``u`` is the minimal change to ``rel_vel`` that places it
    exactly on the VO boundary. The feasible-side normal ``n`` is *not*
    simply ``u`` normalized -- that only coincides with the correct normal
    when ``rel_vel`` starts out inside the VO; when it starts outside
    (``u`` points inward, back toward the VO) the sign is flipped. Following
    RVO2's convention exactly: for the two disc branches (cutoff-circle
    projection and the colliding fallback) ``n = w / ||w||``; for the leg
    branches ``n`` is the *left* normal of the leg direction (RVO2 stores
    ``line.direction`` as the leg direction -- left leg as computed, right
    leg negated -- and the feasible set is the left half-plane of that
    direction, i.e. ``n = [-dir[1], dir[0]]``).

    Parameters
    ----------
    rel_pos : np.ndarray
        Relative position, neighbor minus self, shape (2,).
    rel_vel : np.ndarray
        Relative velocity, self minus neighbor, shape (2,).
    r : float
        Combined (inflated) radius of the two robots.
    tau : float
        ORCA time horizon.
    dt : float
        Simulation time step, used for the colliding-case cutoff.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Correction vector `u` and feasible-side half-plane normal `n`,
        each shape (2,).
    """
    dist_sq = float(rel_pos @ rel_pos)
    r_sq = r * r

    if dist_sq > r_sq:
        inv_tau = 1.0 / tau
        w = rel_vel - inv_tau * rel_pos
        w_len_sq = float(w @ w)
        dot1 = float(w @ rel_pos)

        if dot1 < 0.0 and dot1 * dot1 > r_sq * w_len_sq:
            # Closest exit is through the cutoff disc.
            w_len = math.sqrt(w_len_sq) if w_len_sq > 1e-18 else 1e-9
            unit_w = w / w_len
            u = (r * inv_tau - w_len) * unit_w
            n = unit_w
        else:
            # Closest exit is through one of the two legs.
            leg = math.sqrt(max(dist_sq - r_sq, 0.0))
            det_val = rel_pos[0] * w[1] - rel_pos[1] * w[0]
            if det_val > 0.0:
                direction = (
                    np.array(
                        [
                            rel_pos[0] * leg - rel_pos[1] * r,
                            rel_pos[0] * r + rel_pos[1] * leg,
                        ]
                    )
                    / dist_sq
                )
            else:
                direction = (
                    -np.array(
                        [
                            rel_pos[0] * leg + rel_pos[1] * r,
                            -rel_pos[0] * r + rel_pos[1] * leg,
                        ]
                    )
                    / dist_sq
                )
            u = float(rel_vel @ direction) * direction - rel_vel
            n = np.array([-direction[1], direction[0]])
    else:
        # Already colliding: use the time-step cutoff disc.
        inv_dt = 1.0 / dt
        w = rel_vel - inv_dt * rel_pos
        w_len = float(np.linalg.norm(w))
        unit_w = w / w_len if w_len > 1e-9 else np.array([1.0, 0.0])
        u = (r * inv_dt - w_len) * unit_w
        n = unit_w

    return u, n


class NHORCAAgent(BaseAgent):
    """Reciprocal collision avoidance for a nonholonomic (unicycle) robot.

    Solves a 2-D QP for a preferred holonomic velocity `v_h` subject to
    ORCA half-planes against visible neighbors, a speed-disc bound, and a
    hard trackability cone around the current heading, then tracks `v_h`
    with a heading-alignment + capped-forward-speed unicycle controller.
    """

    def __init__(self, node_id, s0, goal, config, params=None):
        super().__init__(node_id, s0, goal, config, params)
        self.v_h = np.zeros(2)
        # Per-neighbor (step, position) at last sighting, used to gate
        # finite-differenced neighbor velocities against staleness.
        self._prev_neighbor: dict[int, tuple[int, np.ndarray]] = {}
        self._step = 0
        self.infeasible_count = 0

    def compute_input(self, neighbor_states: dict[int, np.ndarray]) -> np.ndarray:
        params = self.params or {}
        tau = params.get('tau', 2.0)
        epsilon = params.get('epsilon', 0.2)
        k_omega = params.get('k_omega', 4.0)
        v_h_max = params.get('v_h_max', 0.95 * self.config.v_max)
        k_pref = params.get('k_pref', 1.0)
        n_disc = params.get('n_disc', 16)

        # This call's step index; used below to detect stale neighbor sightings.
        current_step = self._step
        self._step += 1

        p = self.s[:2]
        theta = self.s[2]
        r = self.config.safety_distance + 2.0 * epsilon

        # Preferred velocity: toward the goal, capped (tapers near the goal).
        v_pref = k_pref * (self.goal - p)
        pref_speed = float(np.linalg.norm(v_pref))
        if pref_speed > v_h_max:
            v_pref = v_pref / pref_speed * v_h_max

        # ORCA half-planes: n . (v - point) >= 0  <=>  (-n) . v <= -(n . point).
        orca_normals = []
        orca_offsets = []
        for j, s_j in neighbor_states.items():
            p_j = s_j[:2]
            prev = self._prev_neighbor.get(j)
            # Only finite-difference against a sighting from exactly the
            # previous call; a neighbor that just re-entered comm range
            # (or was skipped a step) has no reliable prior sample, so
            # fall back to v_j = 0 rather than a spurious huge velocity.
            if prev is not None and prev[0] == current_step - 1:
                v_j = (p_j - prev[1]) / self.config.dt
            else:
                v_j = np.zeros(2)
            self._prev_neighbor[j] = (current_step, p_j.copy())

            rel_pos = p_j - p
            rel_vel = self.v_h - v_j
            u, n = _orca_u(rel_pos, rel_vel, r, tau, self.config.dt)
            u_norm = float(np.linalg.norm(u))
            if u_norm < 1e-9:
                continue
            point = self.v_h + 0.5 * u
            orca_normals.append(n)
            orca_offsets.append(float(n @ point))

        orca_G = np.vstack([-n for n in orca_normals]) if orca_normals else np.zeros((0, 2))
        orca_h = np.array([-off for off in orca_offsets]) if orca_offsets else np.zeros(0)

        # Speed-disc bound ||v|| <= v_h_max, approximated by tangent half-planes:
        # n_k . v <= v_h_max.
        disc_normals = [
            np.array([math.cos(2.0 * math.pi * k / n_disc), math.sin(2.0 * math.pi * k / n_disc)])
            for k in range(n_disc)
        ]
        disc_G = np.vstack(disc_normals) if disc_normals else np.zeros((0, 2))
        disc_h = np.full(n_disc, v_h_max)

        # Trackability cone: restrict v_h to a cone of half-angle e_max around
        # the current heading theta, hard (never relaxed by the ORCA slack
        # fallback below). The unicycle tracking law steers omega = k_omega*e
        # (e = heading error), so the heading realigns with time constant
        # 1/k_omega; while realigning, the lateral velocity error is
        # ||v_h||*sin(e), so the drift accumulated is at most
        # ||v_h||*sin(e)/k_omega. Bounding that by epsilon and solving for e
        # gives sin(e) <= epsilon*k_omega/||v_h||; using v_h_max in place of
        # ||v_h|| gives a conservative, velocity-independent e_max.
        ratio = epsilon * k_omega / v_h_max if v_h_max > 1e-9 else 1.0
        e_max = math.asin(min(1.0, ratio))
        cone_G = np.zeros((0, 2))
        cone_h = np.zeros(0)
        if e_max < math.pi / 2.0 - 1e-9:
            # Cone edges are at theta +/- e_max; the inward normal of each
            # edge ray is that ray's direction rotated by -/+ 90 deg, i.e.
            # the heading rotated by +/-(pi/2 - e_max).
            half_span = math.pi / 2.0 - e_max
            n_plus = np.array([math.cos(theta + half_span), math.sin(theta + half_span)])
            n_minus = np.array([math.cos(theta - half_span), math.sin(theta - half_span)])
            cone_G = np.vstack([-n_plus, -n_minus])
            cone_h = np.zeros(2)
        # else: e_max >= pi/2, the cone covers a full half-plane or more, so
        # it imposes no additional restriction beyond the speed disc.

        G = np.vstack([orca_G, disc_G, cone_G])
        h = np.concatenate([orca_h, disc_h, cone_h])

        P = 2.0 * np.eye(2)
        q = -2.0 * v_pref
        v = qpsolvers.solve_qp(P, q, G, h, solver='quadprog')

        if v is None:
            # Infeasible (dense scenario): relax ORCA half-planes with a shared
            # slack d >= 0, keep the speed-disc bound and trackability cone
            # hard. If this relaxed QP is still infeasible (e.g. the cone
            # and disc, which both always contain v=0, somehow conflict),
            # fall back to v_h = 0 below.
            self.infeasible_count += 1
            n_vars = 3
            P2 = np.diag([2e-3, 2e-3, 2e6])
            q2 = np.array([-2e-3 * v_pref[0], -2e-3 * v_pref[1], 0.0])

            rows = []
            rhs = []
            for n, off in zip(orca_normals, orca_offsets):
                # n . (v - point) >= -d  <=>  (-n) . v - d <= -(n . point).
                rows.append([-n[0], -n[1], -1.0])
                rhs.append(-off)
            for n in disc_normals:
                rows.append([n[0], n[1], 0.0])
                rhs.append(v_h_max)
            for i in range(cone_G.shape[0]):
                rows.append([cone_G[i, 0], cone_G[i, 1], 0.0])
                rhs.append(float(cone_h[i]))
            rows.append([0.0, 0.0, -1.0])
            rhs.append(0.0)

            G2 = np.array(rows) if rows else np.zeros((0, n_vars))
            h2 = np.array(rhs) if rhs else np.zeros(0)
            x = qpsolvers.solve_qp(P2, q2, G2, h2, solver='quadprog')
            v = x[:2] if x is not None else np.zeros(2)

        self.v_h = v

        speed_h = float(np.linalg.norm(self.v_h))
        if speed_h < 1e-6:
            return np.array([0.0, 0.0])

        theta_des = math.atan2(self.v_h[1], self.v_h[0])
        e = _wrap_to_pi(theta_des - theta)
        omega = float(np.clip(k_omega * e, self.config.omega_min, self.config.omega_max))
        v_fwd = float(np.clip(speed_h * math.cos(e), 0.0, self.config.v_max))
        return np.array([v_fwd, omega])


def run_instance(instance: BenchmarkInstance, params: dict | None, out_dir) -> RunResult:
    """Run the NH-ORCA baseline on one benchmark instance.

    Parameters
    ----------
    instance : BenchmarkInstance
        Scenario definition (initial states, goals, config).
    params : dict | None
        NH-ORCA hyperparameters; see `NHORCAAgent` for defaults.
    out_dir
        Accepted for interface compatibility; unused (no artifacts written).
    """
    return run_reactive(NHORCAAgent, instance, params)
