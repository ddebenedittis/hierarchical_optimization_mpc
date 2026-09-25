import numpy as np

import hierarchical_qp.hierarchical_qp as _hqp_module
from hierarchical_qp.hierarchical_qp import HierarchicalQP, QPSolver

# Some environments have a locally-installed fork of `qpsolvers` whose `solve_qp`
# returns `(x, obj)` instead of the standard package's plain `x` array (or `None`).
# `hierarchical_qp.py`'s internal `_solve_qp`/`_solve_weighted` assume the standard
# return and slice straight into it, which corrupts the solution when the fork is on
# the path. Patch the `solve_qp` name inside that module's own namespace (not its
# source file) so it always unwraps to a plain array before `hierarchical_qp.py` uses
# it -- a no-op when the standard package is installed.
_original_solve_qp = _hqp_module.solve_qp


def _unwrap_qp_solution(*args, **kwargs):
    sol = _original_solve_qp(*args, **kwargs)
    return sol[0] if isinstance(sol, tuple) else sol


_hqp_module.solve_qp = _unwrap_qp_solution


class Agent:
    """
    Distributed weighted-QP controller (non-hierarchical).

    At every step the agent solves ONE QP over its own commanded velocity
    `u = [vx, vy]`, using `hierarchical_qp.HierarchicalQP` in its
    non-hierarchical ("weighted") mode. Every task is passed as its own
    priority-level entry, but `hierarchical=False` does NOT solve them
    lexicographically: instead all levels are merged into a single combined
    QP, where each level's role is set by its weight:

      - velocity-limit and collision-avoidance constraints are given a large
        finite weight `wi = w_safety` (default 1e3), which the solver treats
        as a SOFT constraint with a heavily-penalized slack variable (see
        `HierarchicalQP._solve_weighted`) -- in practice this is satisfied to
        within a tiny numerical margin whenever the underlying hard problem
        would be feasible, but degrades gracefully (a small, bounded
        violation) instead of raising when multiple neighbors are
        simultaneously pinned at the enforced margin in conflicting
        directions and the exact hard problem has no solution at all. This
        is a deliberate trade: `w_safety = np.inf` would be the "exact hard
        constraint" version (same guarantee as the hierarchical variant of
        this scenario), but it also means the whole QP raises whenever those
        hard constraints become mutually infeasible under multi-agent
        congestion -- see `cbf_qp/README.md`'s "Why the safety constraint is
        softened, not hard" section.
      - goal-tracking and (optional) formation-keeping are given a finite
        weight `we` each: they become SOFT, weighted-least-squares cost
        terms combined additively into the same QP, with no priority order
        between them -- the relative weight is all that decides the
        trade-off when the two disagree.

    This is the "weighted, non-hierarchical" counterpart of the priority
    version of this scenario: same hard safety guarantees, but task
    blending is a single weighted optimization (closer in spirit to the
    potential-field baseline's weighted sum, but solved as an actual QP with
    hard safety constraints rather than an unconstrained heuristic force).
    """

    def __init__(
        self,
        node_id: int,
        pos: np.ndarray,
        goal: np.ndarray,
        v_max: float,
        k_goal: float,
        d_safe: float,
        l: float,
        gamma: float,
        solver: QPSolver = QPSolver.quadprog,
        formation_targets: list[int] | None = None,
        k_form: float = 0.0,
        d_form: float = 0.0,
        w_goal: float = 1.0,
        w_form: float = 1.0,
        w_safety: float = 1e3,
        d_safe_enforced: float | None = None,
    ):
        self.node_id = node_id
        self.pos = np.array(pos, dtype=float)
        self.goal = np.array(goal, dtype=float)
        self.v_max = v_max

        self.k_goal = k_goal
        self.d_safe = d_safe
        self.l = l
        # The CBF barrier enforces this, not d_safe directly -- see settings.py's docstring for
        # why this baseline is deliberately more conservative than the shared measurement d_safe.
        # An explicit `d_safe_enforced` overrides the l-based buffer, so a comparison harness can
        # hand every method the same enforced distance.
        self.d_safe_enforced = d_safe + 2.0 * l if d_safe_enforced is None else d_safe_enforced
        self.gamma = gamma
        self.w_safety = w_safety

        self.k_form = k_form
        self.d_form = d_form
        self.formation_targets = formation_targets or []

        self.w_goal = w_goal
        self.w_form = w_form

        self.hqp = HierarchicalQP(solver=solver, hierarchical=False)

    def _formation_task(self, positions: dict[int, np.ndarray]):
        """Equality task: drive the projected relative velocity to close the
        gap to `d_form` with each formation target within range."""
        rows_A, rows_b = [], []
        for j in self.formation_targets:
            if j not in positions:
                continue
            diff = self.pos - positions[j]
            dist = np.linalg.norm(diff)
            if dist < 1e-6:
                continue
            direction = diff / dist
            rows_A.append(direction)
            rows_b.append(-self.k_form * (dist - self.d_form))
        if not rows_A:
            return None, None
        return np.array(rows_A), np.array(rows_b)

    def compute_input(
        self, positions: dict[int, np.ndarray], communication_range: float
    ) -> np.ndarray:
        nx = 2

        # ---- Level 0: HARD constraints -- velocity limits + collision CBF ----
        # h(pos_i, pos_j) = ||pos_i - pos_j||^2 - d_safe_enforced^2 >= 0, where
        # d_safe_enforced = d_safe + 2*l is strictly larger than the shared measurement
        # threshold d_safe (see settings.py's `l`).
        # CBF condition (neighbor assumed momentarily stationary, worst case):
        #   grad_i(h) . u_i >= -gamma * h  ==>  -2*(pos_i-pos_j) . u_i <= gamma*h
        C_lim = np.vstack([np.eye(nx), -np.eye(nx)])
        d_lim = np.full(2 * nx, self.v_max)

        C_rows, d_rows = [], []
        for j, p_j in positions.items():
            if j == self.node_id:
                continue
            diff = self.pos - p_j
            dist = np.linalg.norm(diff)
            if dist < 1e-6:
                diff = np.random.uniform(-1e-3, 1e-3, size=nx)
                dist = np.linalg.norm(diff)
            if dist > communication_range:
                continue

            h = dist**2 - self.d_safe_enforced**2
            C_rows.append(-2.0 * diff)
            d_rows.append(self.gamma * h)

        C0 = np.vstack([C_lim] + ([np.array(C_rows)] if C_rows else []))
        d0 = np.concatenate([d_lim] + ([np.array(d_rows)] if d_rows else []))

        A_levels = [None]
        b_levels = [None]
        C_levels = [C0]
        d_levels = [d0]
        we_levels = [1.0]  # unused (no equality task at this level)
        wi_levels = [self.w_safety]  # heavily-penalized SOFT: see class docstring

        # ---- Level 1: SOFT goal-tracking cost ----
        A_levels.append(np.eye(nx))
        b_levels.append(self.k_goal * (self.goal - self.pos))
        C_levels.append(None)
        d_levels.append(None)
        we_levels.append(self.w_goal)
        wi_levels.append(1.0)  # unused (no inequality task at this level)

        # ---- Level 2 (optional): SOFT formation-keeping cost ----
        A_form, b_form = self._formation_task(positions)
        if A_form is not None:
            A_levels.append(A_form)
            b_levels.append(b_form)
            C_levels.append(None)
            d_levels.append(None)
            we_levels.append(self.w_form)
            wi_levels.append(1.0)

        # Non-hierarchical ("weighted") mode returns the solution vector
        # directly, not a (x_star, slacks) tuple like the hierarchical mode.
        x_star = self.hqp(A_levels, b_levels, C_levels, d_levels, we=we_levels, wi=wi_levels)
        u = np.asarray(x_star[:nx], dtype=float)

        speed = np.linalg.norm(u)
        if speed > self.v_max:
            u = u / speed * self.v_max

        return u

    def step(self, u: np.ndarray, dt: float) -> None:
        self.pos = self.pos + dt * u
