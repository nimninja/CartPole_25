"""Map Arduino encoder counts to sim-like [x, x_dot, theta, theta_dot]."""

from __future__ import annotations

import math

import numpy as np

# Pole upright ≈ this count (old reward band was ~500–700)
UPRIGHT_ANGLE = 600
ANGLE_WRAP = 1200
BELT_PER_SIM_X = 5800.0


def _dist_from_upright_counts(angle: float) -> float:
    """0 at upright, up to ANGLE_WRAP/2 at opposite side."""
    a = float(angle)
    d = abs(a - UPRIGHT_ANGLE)
    return min(d, ANGLE_WRAP - d)


def angle_count_to_theta(angle: float) -> float:
    """
    0 rad at upright (count ~600), pi rad when hanging (count ~0).
    Matches sim reset at theta=pi so SB3 clipping behaves like sim training.
    """
    dist = _dist_from_upright_counts(angle)
    return float(math.pi * min(dist / UPRIGHT_ANGLE, 1.0))


def unwrap_count_delta(new: float, old: float, wrap: int = ANGLE_WRAP) -> float:
    d = float(new) - float(old)
    half = wrap / 2.0
    if d > half:
        d -= wrap
    elif d < -half:
        d += wrap
    return d


def physics_observation_from_state(
    angle: float,
    belt: float,
    prev_angle: float,
    prev_belt: float,
    prev_theta: float,
    prev_x: float,
    step_dt: float,
) -> np.ndarray:
    theta = angle_count_to_theta(angle)
    x = float(belt) / BELT_PER_SIM_X

    da = unwrap_count_delta(angle, prev_angle)
    theta_dot = (theta - prev_theta) / step_dt
    x_dot = (x - prev_x) / step_dt

    return np.array([x, x_dot, theta, theta_dot], dtype=np.float32), theta, x, da


def physics_observation(
    angle: float,
    belt: float,
    ang_vel: float,
    belt_vel: float,
) -> np.ndarray:
    """Legacy path; prefer physics_observation_from_state with fixed step_dt."""
    theta = angle_count_to_theta(angle)
    x = float(belt) / BELT_PER_SIM_X
    x_dot = float(belt_vel) / BELT_PER_SIM_X
    theta_dot = float(ang_vel) / (UPRIGHT_ANGLE / math.pi)
    return np.array([x, x_dot, theta, theta_dot], dtype=np.float32)
