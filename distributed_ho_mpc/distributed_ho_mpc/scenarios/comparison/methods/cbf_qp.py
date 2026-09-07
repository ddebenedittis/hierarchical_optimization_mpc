"""Robotarium-style unicycle CBF-QP baseline controller.

Implements a per-step control barrier function (CBF) quadratic program for
differential-drive (unicycle) robots, following the Georgia Tech Robotarium
formulation: the unicycle is mapped to a single-integrator offset point via
a near-identity diffeomorphism, a nominal single-integrator controller drives
that point toward the goal, and pairwise CBF constraints (cubic class-K,
reciprocal responsibility split) enforce collision avoidance as hard
constraints in the QP.
"""

from __future__ import annotations

import numpy as np
from qpsolvers import solve_qp

from distributed_ho_mpc.scenarios.comparison.common.base_agent import BaseAgent, run_reactive
from distributed_ho_mpc.scenarios.comparison.common.benchmark import (
    BenchmarkConfig,
    BenchmarkInstance,
)
from distributed_ho_mpc.scenarios.comparison.common.run_io import RunResult


def _wrap_angle(angle: float) -> float:
    """Wrap an angle (rad) to [-pi, pi]."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


class CBFQPAgent(BaseAgent):
    """Unicycle CBF-QP controller (Robotarium barrier certificate).

    At every step:
      1. The unicycle pose is mapped to an offset point `p_hat` ahead of the
         robot center via a near-identity diffeomorphism (parameter `l`).
      2. A nominal single-integrator controller drives `p_hat` toward the
         goal (proportional gain `k_goal`, speed-capped).
      3. Pairwise CBF constraints (cubic class-K, gain `gamma`) enforce
         `||p_hat_i - p_hat_j|| >= d_safe` as hard linear constraints on the
         unicycle input `[v, omega]`. The 1/2 factor on the barrier gain is
         the reciprocal responsibility split: both agents run the same
         controller and solve the same constraint independently, so each is
         only responsible for half of the required correction.
      4. The QP projects the nominal input onto the feasible set (CBF rows
         plus the hard input box), minimizing a weighted distance to the
         nominal input. No post-solve clipping is applied.
      5. If the hard QP is infeasible (possible when `h < 0`, i.e. already
         inside another agent's safety margin), it is re-solved with a
         single shared slack variable added to the CBF rows only; if that
         also fails, a safe fallback (rotate in place toward the goal) is
         returned.

    Parameters (read from `params`, defaults shown)
    -------------------------------------------------
    l : float, default 0.3
        Near-identity projection distance (offset point ahead of center).
    gamma : float, default 1.0
        CBF barrier gain (cubic class-K).
    k_goal : float, default 1.0
        Nominal single-integrator controller gain.
    w_omega : float | None, default None (resolved to `l**2`)
        Weight on omega in the QP tracking cost.
    slack_penalty : float, default 1e6
        Quadratic penalty on the shared slack variable used when the hard
        QP is infeasible.
    """

    def __init__(
        self,
        node_id: int,
        s0: np.ndarray,
        goal: np.ndarray,
        config: BenchmarkConfig,
        params: dict | None = None,
    ) -> None:
        super().__init__(node_id, s0, goal, config, params)
        p = self.params or {}
        self.l = p.get('l', 0.3)
        self.gamma = p.get('gamma', 1.0)
        self.k_goal = p.get('k_goal', 1.0)
        self.w_omega = p.get('w_omega', None)
        if self.w_omega is None:
            self.w_omega = self.l**2
        self.slack_penalty = p.get('slack_penalty', 1e6)
        self.infeasible_count = 0
        self._rng = np.random.default_rng(node_id)

    def compute_input(self, neighbor_states: dict[int, np.ndarray]) -> np.ndarray:
        """Solve the per-step CBF-QP and return `[v, omega]`."""
        cfg = self.config
        # The barrier guards the offset points p_hat (l ahead of center), whose
        # separation can exceed the center-to-center distance by up to 2*l
        # depending on heading. Inflate the barrier radius by 2*l so that
        # ||p_hat_i - p_hat_j|| >= d_cbf guarantees ||p_i - p_j|| >= safety_distance
        # for any heading (Robotarium single-integrator-to-unicycle convention).
        d_cbf = cfg.safety_distance + 2.0 * self.l

        x, y, theta = self.s
        c, sn = np.cos(theta), np.sin(theta)
        p = np.array([x, y])
        p_hat = p + self.l * np.array([c, sn])
        J = np.array([[c, -self.l * sn], [sn, self.l * c]])
        J_inv = np.array([[c, sn], [-sn / self.l, c / self.l]])

        v_si = self.k_goal * (self.goal - p_hat)
        speed = np.linalg.norm(v_si)
        v_max_soft = 0.95 * cfg.v_max
        if speed > v_max_soft:
            v_si = v_si / speed * v_max_soft
        u_nom = J_inv @ v_si

        # ---- CBF rows: -2 * diff^T @ J @ u <= (gamma / 2) * h**3 ----
        G_cbf, h_cbf = [], []
        for pose_j in neighbor_states.values():
            xj, yj, theta_j = pose_j
            p_hat_j = np.array([xj, yj]) + self.l * np.array([np.cos(theta_j), np.sin(theta_j)])
            diff = p_hat - p_hat_j
            dist = np.linalg.norm(diff)
            if dist < 1e-6:
                diff = diff + self._rng.uniform(-1e-3, 1e-3, size=2)
                dist = np.linalg.norm(diff)
            h = dist**2 - d_cbf**2
            G_cbf.append(-2.0 * (diff @ J))
            h_cbf.append(0.5 * self.gamma * h**3)

        # ---- Hard box rows: v_min <= v <= v_max, omega_min <= omega <= omega_max ----
        G_box = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
        h_box = np.array([cfg.v_max, -cfg.v_min, cfg.omega_max, -cfg.omega_min])

        W = np.diag([1.0, self.w_omega])
        P = 2.0 * W
        q = -2.0 * W @ u_nom

        if G_cbf:
            G = np.vstack([np.array(G_cbf), G_box])
            h_vec = np.concatenate([np.array(h_cbf), h_box])
        else:
            G, h_vec = G_box, h_box

        z = solve_qp(P, q, G, h_vec, solver='quadprog')

        if z is None:
            z = self._solve_with_slack(P, q, G_cbf, h_cbf, G_box, h_box)
            self.infeasible_count += 1

        if z is None:
            angle_to_goal = np.arctan2(self.goal[1] - y, self.goal[0] - x)
            heading_error = _wrap_angle(angle_to_goal - theta)
            omega = np.clip(self.k_goal * heading_error, cfg.omega_min, cfg.omega_max)
            return np.array([0.0, omega])

        return np.asarray(z[:2], dtype=float)

    def _solve_with_slack(
        self,
        P: np.ndarray,
        q: np.ndarray,
        G_cbf: list[np.ndarray],
        h_cbf: list[float],
        G_box: np.ndarray,
        h_box: np.ndarray,
    ) -> np.ndarray | None:
        """Re-solve with a single shared slack `delta >= 0` on the CBF rows only."""
        P_s = np.zeros((3, 3))
        P_s[:2, :2] = P
        P_s[2, 2] = 2.0 * self.slack_penalty
        q_s = np.concatenate([q, [0.0]])

        rows_G, rows_h = [], []
        for row, h_val in zip(G_cbf, h_cbf):
            rows_G.append(np.concatenate([row, [-1.0]]))
            rows_h.append(h_val)
        for row, h_val in zip(G_box, h_box):
            rows_G.append(np.concatenate([row, [0.0]]))
            rows_h.append(h_val)
        rows_G.append(np.array([0.0, 0.0, -1.0]))  # delta >= 0
        rows_h.append(0.0)

        G_s = np.array(rows_G)
        h_s = np.array(rows_h)

        z_s = solve_qp(P_s, q_s, G_s, h_s, solver='quadprog')
        if z_s is None:
            return None
        return z_s[:2]


def run_instance(instance: BenchmarkInstance, params: dict | None, out_dir) -> RunResult:
    """Entry point for the comparison campaign driver."""
    return run_reactive(CBFQPAgent, instance, params)
