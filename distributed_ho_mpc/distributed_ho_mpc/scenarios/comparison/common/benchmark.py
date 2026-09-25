"""Seeded benchmark instance generation for the radial-switching comparison.

Replicates the instance generator in
``radial_switching_unicycle/network_simulation.py`` (lines ~229-259), but
seeded via ``numpy.random.default_rng`` so campaigns are reproducible.

N robots sit on a circle of radius ``radius``. Each robot's goal is a point
on the circle and its start is the antipodal point, so every robot's path
crosses the center. Robots start already facing (approximately) their goal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BenchmarkConfig:
    """Shared parameters for a radial-switching instance and its simulation."""

    n_robots: int = 8
    radius: float = 10.0
    min_chord: float = 2.6
    heading_perturbation: float = np.pi / 6
    dt: float = 0.05
    max_steps: int = 600
    safety_distance: float = 2.0
    v_min: float = -1.6
    v_max: float = 1.6
    omega_min: float = -2.0
    omega_max: float = 2.0
    goal_tol: float = 0.1
    comm_range: float = 5.0
    limit_connection: int = 7


@dataclass
class BenchmarkInstance:
    """One radial-switching problem instance: initial states, goals, and config."""

    seed: int  # -1 for the deterministic symmetric instance
    s_init: np.ndarray  # (N, 3) [x, y, theta]
    goals: np.ndarray  # (N, 2)
    config: BenchmarkConfig


def discretization_back_off(config: BenchmarkConfig) -> float:
    """Constraint tightening for a collision bound enforced only at sample instants.

    The dHQP/dWQP collision task is a linearized inequality imposed on the
    predicted state at the next sample. Between samples the controller cannot
    react, so a closing pair can cross the bound by up to one sample of relative
    displacement, ``2 * v_max * dt``. Measured undershoot tracks this closely and
    scales linearly with ``dt`` (median 0.155 m at dt = 0.05, 0.298 m at
    dt = 0.10, against bounds of 0.160 and 0.320), confirming the mechanism is
    discretization rather than the hierarchy, the distribution, or the horizon.

    Tightening by this amount is the same kind of allowance the reactive
    baselines already build in -- CBF-QP guards its offset points at
    ``d_safe + 2*l`` and NH-ORCA inflates its radius by ``2*epsilon`` -- so it
    puts every method on the same footing while all of them are still scored
    against the common ``config.safety_distance`` contract.

    Derived rather than tuned on purpose: a margin fitted until a seed batch
    comes out clean is not predictive and will break on the next seed, whereas
    this one correctly anticipates the larger back-off needed at a slower
    control rate without re-tuning.

    Args:
        config: Benchmark configuration supplying ``v_max`` and ``dt``.

    Returns:
        The back-off in meters, to be added to ``config.safety_distance``.
    """
    return 2.0 * config.v_max * config.dt


def _sample_angles(n_robots: int, min_angle_sep: float, rng: np.random.Generator) -> list[float]:
    """Rejection-sample ``n_robots`` angles in [0, 2*pi) with minimum circular separation."""
    angles: list[float] = []
    while len(angles) < n_robots:
        candidate = rng.uniform(0, 2 * np.pi)
        if all(
            min(abs(candidate - a), 2 * np.pi - abs(candidate - a)) >= min_angle_sep for a in angles
        ):
            angles.append(candidate)
    return angles


def _build_instance(
    angles: list[float],
    perturbations: list[float],
    seed: int,
    config: BenchmarkConfig,
) -> BenchmarkInstance:
    """Build a BenchmarkInstance from per-robot goal angles and heading perturbations."""
    center = np.array([0.0, 0.0])
    goals = []
    s_init = []
    for theta, perturbation in zip(angles, perturbations):
        goal = center + config.radius * np.array([np.cos(theta), np.sin(theta)])
        goals.append(goal)

        theta_init = theta - np.pi
        pos_init = center + config.radius * np.array([np.cos(theta_init), np.sin(theta_init)])
        heading = theta_init + np.pi + perturbation  # == theta + perturbation
        s_init.append(np.array([pos_init[0], pos_init[1], heading]))

    return BenchmarkInstance(
        seed=seed,
        s_init=np.array(s_init),
        goals=np.array(goals),
        config=config,
    )


def generate_instance(seed: int, config: BenchmarkConfig | None = None) -> BenchmarkInstance:
    """Generate a seeded random radial-switching instance.

    Robots' goal angles are rejection-sampled uniformly on the circle so
    every pair has circular angular separation
    >= ``2 * arcsin(min_chord / (2 * radius))``. Each robot starts at the
    antipodal point on the circle, initially heading toward its goal plus a
    uniform random perturbation in
    ``[-heading_perturbation, heading_perturbation]``.

    Args:
        seed: Seed for ``numpy.random.default_rng``.
        config: Benchmark configuration; defaults to ``BenchmarkConfig()``.

    Returns:
        The generated instance.
    """
    config = config or BenchmarkConfig()
    rng = np.random.default_rng(seed)
    min_angle_sep = 2 * np.arcsin(config.min_chord / (2 * config.radius))
    angles = _sample_angles(config.n_robots, min_angle_sep, rng)
    perturbations = [
        rng.uniform(-config.heading_perturbation, config.heading_perturbation)
        for _ in range(config.n_robots)
    ]
    return _build_instance(angles, perturbations, seed, config)


def symmetric_instance(config: BenchmarkConfig | None = None) -> BenchmarkInstance:
    """Build the deterministic symmetric instance: robots evenly spaced on the circle.

    Args:
        config: Benchmark configuration; defaults to ``BenchmarkConfig()``.

    Returns:
        The symmetric instance, with ``seed == -1``.
    """
    config = config or BenchmarkConfig()
    angles = [2 * np.pi * i / config.n_robots for i in range(config.n_robots)]
    perturbations = [0.0] * config.n_robots
    return _build_instance(angles, perturbations, -1, config)
