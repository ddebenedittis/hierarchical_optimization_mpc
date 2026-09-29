import numpy as np


class Agent:
    """
    Minimal reactive controller.

    At every step the commanded velocity is the sum of a goal-attraction
    term, a pairwise collision-repulsion term, and (optionally) a
    formation-keeping spring term, combined through fixed scalar weights.
    There is no priority structure between these terms: this is the "just
    use something simple" behaviour-based/potential-field baseline, meant to
    be contrasted against dHQP's strict task hierarchy.
    """

    def __init__(
        self,
        node_id: int,
        pos: np.ndarray,
        goal: np.ndarray,
        v_max: float,
        k_goal: float,
        k_rep: float,
        d_safe: float,
        formation_targets: list[int] | None = None,
        k_form: float = 0.0,
        d_form: float = 0.0,
    ):
        self.node_id = node_id
        self.pos = np.array(pos, dtype=float)
        self.goal = np.array(goal, dtype=float)
        self.v_max = v_max

        self.k_goal = k_goal
        self.k_rep = k_rep
        self.d_safe = d_safe

        self.k_form = k_form
        self.d_form = d_form
        self.formation_targets = formation_targets or []

    def compute_input(
        self, positions: dict[int, np.ndarray], communication_range: float
    ) -> np.ndarray:
        u = self.k_goal * (self.goal - self.pos)

        for j, p_j in positions.items():
            if j == self.node_id:
                continue

            diff = self.pos - p_j
            dist = np.linalg.norm(diff)
            if dist < 1e-6:
                diff = np.random.uniform(-1e-3, 1e-3, size=2)
                dist = np.linalg.norm(diff)
            if dist > communication_range:
                continue

            if dist < self.d_safe:
                u += self.k_rep * (1.0 / dist - 1.0 / self.d_safe) / dist**2 * (diff / dist)

            if j in self.formation_targets:
                u += self.k_form * (dist - self.d_form) * (-diff / dist)

        speed = np.linalg.norm(u)
        if speed > self.v_max:
            u = u / speed * self.v_max

        return u

    def step(self, u: np.ndarray, dt: float) -> None:
        # Plant saturation ||u|| <= v_max, applied identically by every method in the
        # comparison regardless of what its controller already guarantees.
        u = np.asarray(u, dtype=float)
        speed = np.linalg.norm(u)
        if speed > self.v_max:
            u = u / speed * self.v_max
        self.u_applied = u
        self.pos = self.pos + dt * u
