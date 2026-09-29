"""Shared unicycle integrator, reactive-agent base class, and sim loop.

The integrator faithfully ports the state propagation the dHQP
``radial_switching_unicycle`` scenario actually simulates with (``Node.evolve``
in ``node.py``): plain forward-Euler substepping, not the midpoint-heading
model in ``robot_models.get_unicycle_model`` (that model is only used
internally by the HO-MPC solver to predict its horizon).
"""

from __future__ import annotations

import time

import numpy as np

from distributed_ho_mpc.scenarios.comparison.common.benchmark import (
    BenchmarkConfig,
    BenchmarkInstance,
)
from distributed_ho_mpc.scenarios.comparison.common.run_io import RunResult


def clamp_input(u: np.ndarray, config: BenchmarkConfig) -> np.ndarray:
    """Clip a [v, omega] input to the configured actuation box.

    Args:
        u: Input [v, omega].
        config: Benchmark configuration providing v_min/v_max, omega_min/omega_max.

    Returns:
        The clipped [v, omega] input.
    """
    return np.array(
        [
            np.clip(u[0], config.v_min, config.v_max),
            np.clip(u[1], config.omega_min, config.omega_max),
        ]
    )


def unicycle_step(s: np.ndarray, u: np.ndarray, dt: float, n_substeps: int = 10) -> np.ndarray:
    """Advance a unicycle state by dt via forward-Euler substepping.

    Faithful numpy port of ``Node.evolve`` in
    ``radial_switching_unicycle/node.py``: the step ``dt`` is split into
    ``n_substeps`` equal sub-intervals (matching the reference's
    ``n_intervals = 10``, applied there to the *unscaled* simulation
    ``dt``), and each sub-interval applies plain forward Euler::

        x += (dt / n_substeps) * v * cos(theta)
        y += (dt / n_substeps) * v * sin(theta)
        theta += (dt / n_substeps) * omega

    with theta re-evaluated after every sub-interval. This is distinct from
    the midpoint-heading discretization in
    ``robot_models.get_unicycle_model`` (``theta + dt/2 * omega`` inside the
    cos/sin), which that reference only uses as the HO-MPC solver's internal
    prediction model, not to integrate the true simulated trajectory.

    Args:
        s: Current state [x, y, theta].
        u: Input [v, omega].
        dt: Total step duration.
        n_substeps: Number of forward-Euler sub-intervals within dt.

    Returns:
        The state after dt.
    """
    s = np.array(s, dtype=float, copy=True)
    v, omega = u[0], u[1]
    h = dt / n_substeps
    for _ in range(n_substeps):
        s = s + h * np.array([v * np.cos(s[2]), v * np.sin(s[2]), omega])
    return s


class BaseAgent:
    """Base class for a reactive radial-switching agent.

    Subclasses implement ``compute_input`` to map the agent's own state and
    the (already comm-range-gated) neighbor states to a control input.
    """

    def __init__(
        self,
        node_id: int,
        s0: np.ndarray,
        goal: np.ndarray,
        config: BenchmarkConfig,
        params: dict | None = None,
    ) -> None:
        self.node_id = node_id
        self.s = np.array(s0, dtype=float, copy=True)
        self.goal = np.array(goal, dtype=float, copy=True)
        self.config = config
        self.params = params if params is not None else {}

    def compute_input(self, neighbor_states: dict[int, np.ndarray]) -> np.ndarray:
        """Compute a [v, omega] control input given neighbor poses in comm range.

        Args:
            neighbor_states: Map from neighbor id to its (3,) pose
                [x, y, theta], already filtered to neighbors within
                ``self.config.comm_range`` of this agent.

        Returns:
            The [v, omega] input (before clamping).
        """
        raise NotImplementedError

    def step(self, u: np.ndarray, dt: float) -> None:
        """Advance this agent's state by dt under input u."""
        self.s = unicycle_step(self.s, u, dt)


def run_reactive(
    agent_cls: type,
    instance: BenchmarkInstance,
    params: dict | None = None,
    max_steps: int | None = None,
) -> RunResult:
    """Run a generic synchronous reactive-agent simulation loop.

    Builds one ``agent_cls`` instance per robot, then repeatedly: gates each
    agent's view of the other robots' poses by Euclidean distance
    <= ``config.comm_range``, calls ``compute_input`` (timed with
    ``time.perf_counter`` into ``solve_times``), clamps the result with
    ``clamp_input``, and finally steps every agent synchronously from the
    same pre-step state snapshot (all inputs are computed before any agent
    moves). Stops early once every robot is within ``config.goal_tol`` of
    its goal, truncating the recorded history.

    Args:
        agent_cls: A ``BaseAgent`` subclass.
        instance: The problem instance (initial states, goals, config).
        params: Optional params dict forwarded to every agent's constructor.
        max_steps: Override ``instance.config.max_steps``.

    Returns:
        The recorded RunResult (trajectories, inputs, solve times,
        infeasible_count summed from any per-agent ``infeasible_count``
        attribute).
    """
    config = instance.config
    n_robots = config.n_robots
    steps = max_steps if max_steps is not None else config.max_steps
    params = params if params is not None else {}

    agents = [
        agent_cls(i, instance.s_init[i], instance.goals[i], config, params) for i in range(n_robots)
    ]

    x_hist = np.zeros((steps + 1, n_robots, 3))
    u_hist = np.zeros((steps, n_robots, 2))
    solve_times = np.zeros((steps, n_robots))

    x_hist[0] = np.array([agent.s for agent in agents])

    final_step = steps
    for t in range(steps):
        snapshot = np.array([agent.s for agent in agents])
        inputs = np.zeros((n_robots, 2))

        for i, agent in enumerate(agents):
            neighbor_states = {
                j: snapshot[j]
                for j in range(n_robots)
                if j != i and np.linalg.norm(snapshot[j, :2] - snapshot[i, :2]) <= config.comm_range
            }
            t0 = time.perf_counter()
            u = agent.compute_input(neighbor_states)
            solve_times[t, i] = time.perf_counter() - t0
            inputs[i] = clamp_input(u, config)

        for i, agent in enumerate(agents):
            agent.step(inputs[i], config.dt)

        x_hist[t + 1] = np.array([agent.s for agent in agents])
        u_hist[t] = inputs

        reached = np.linalg.norm(x_hist[t + 1, :, :2] - instance.goals, axis=1) <= config.goal_tol
        if np.all(reached):
            final_step = t + 1
            break

    x_hist = x_hist[: final_step + 1]
    u_hist = u_hist[:final_step]
    solve_times = solve_times[:final_step]

    infeasible_count = sum(getattr(agent, 'infeasible_count', 0) for agent in agents)

    return RunResult(
        x_hist=x_hist,
        u_hist=u_hist,
        solve_times=solve_times,
        infeasible_count=infeasible_count,
    )
