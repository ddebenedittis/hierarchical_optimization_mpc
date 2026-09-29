import numpy as np

from distributed_ho_mpc.scenarios.orca_radial_switching import orca


class Agent:
    """
    Minimal ORCA agent.

    Each step it computes a preferred velocity (simple proportional
    go-to-goal law), then corrects it against every neighbour within
    `communication_range` using the reciprocal ORCA half-plane construction
    in `orca.py`, and finally integrates its position with the corrected
    velocity. There is no shared/joint optimization problem, no task
    hierarchy, and no consensus/dual-variable exchange between agents --
    the collision-avoidance guarantee comes entirely from the ORCA
    reciprocity assumption (both agents apply the same rule).
    """

    def __init__(
        self,
        node_id: int,
        pos: np.ndarray,
        goal: np.ndarray,
        v_max: float,
        k_goal: float,
        orca_radius: float,
        orca_time_horizon: float,
        orca_max_speed: float,
        dt: float,
    ):
        self.node_id = node_id
        self.pos = np.array(pos, dtype=float)
        self.goal = np.array(goal, dtype=float)
        self.v_max = v_max
        self.k_goal = k_goal

        self.orca_radius = orca_radius
        self.orca_time_horizon = orca_time_horizon
        self.orca_max_speed = orca_max_speed
        self.dt = dt

        self.velocity = np.zeros(2)

    def preferred_velocity(self) -> np.ndarray:
        """Straight-line go-to-goal velocity, saturated to v_max."""
        u = self.k_goal * (self.goal - self.pos)
        speed = np.linalg.norm(u)
        if speed > self.v_max:
            u = u / speed * self.v_max
        return u

    def compute_input(
        self,
        positions: dict[int, np.ndarray],
        velocities: dict[int, np.ndarray],
        communication_range: float,
    ) -> np.ndarray:
        """ORCA-corrected velocity against every sensed neighbour."""
        neighbors = [
            (p_j, velocities[j], self.orca_radius)
            for j, p_j in positions.items()
            if j != self.node_id and np.linalg.norm(self.pos - p_j) <= communication_range
        ]

        return orca.compute_new_velocity(
            position=self.pos,
            pref_velocity=self.preferred_velocity(),
            neighbors=neighbors,
            radius=self.orca_radius,
            time_horizon=self.orca_time_horizon,
            max_speed=self.orca_max_speed,
            dt=self.dt,
            velocity=self.velocity,
        )

    def step(self, u: np.ndarray, dt: float) -> None:
        self.velocity = np.array(u, dtype=float)
        self.pos = self.pos + dt * self.velocity
