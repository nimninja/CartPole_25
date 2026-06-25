"""
LQR balance for inverted cart-pole (balance region).

Standard hybrid from Furuta et al. (1992) / swing-up literature:
RL or energy pumping for swing-up, LQR when |theta| is small.

State (small-angle): [x, x_dot, theta, theta_dot]
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(raw) if raw is not None else default


@dataclass
class BalanceCommand:
    action: int  # 0=LEFT, 1=RIGHT, 2=STOP
    pulse_ms: int
    force: float


def _cartpole_linearized(
    mc: float, mp: float, half_length: float, gravity: float = 9.8
) -> tuple[np.ndarray, np.ndarray]:
    total = mc + mp
    polemass_length = mp * half_length
    denom = half_length * (4.0 / 3.0 - mp / total)

    a_theta = gravity / denom
    b_theta = -1.0 / (total * denom)
    b_x = 1.0 / total - polemass_length * b_theta / total
    a_x_theta = polemass_length * a_theta / total

    a = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, -a_x_theta, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, a_theta, 0.0],
        ],
        dtype=np.float64,
    )
    b = np.array([[0.0], [b_x], [0.0], [b_theta]], dtype=np.float64)
    return a, b


def _discretize_euler(a: np.ndarray, b: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    n = a.shape[0]
    return np.eye(n) + a * dt, b * dt


def _solve_discrete_lqr(
    ad: np.ndarray,
    bd: np.ndarray,
    q: np.ndarray,
    r: np.ndarray,
    max_iter: int = 2000,
    tol: float = 1e-12,
) -> np.ndarray:
    """Discrete-time LQR via DARE iteration."""
    p = q.copy()
    r_inv = np.linalg.inv(r)
    bt = bd.T
    at = ad.T
    for _ in range(max_iter):
        bt_p = bt @ p
        s = r + bt_p @ bd
        k = np.linalg.solve(s, bt_p @ ad)
        p_new = q + at @ p @ ad - at @ p @ bd @ k
        if np.max(np.abs(p_new - p)) < tol:
            p = p_new
            break
        p = p_new
    return np.linalg.solve(r + bt @ p @ bd, bt @ p @ ad).reshape(1, 4)


def default_lqr_gain(
    mc: float = 1.0,
    mp: float = 0.2,
    half_length: float = 0.5,
    dt: float = 0.01,
    q_diag: tuple[float, float, float, float] = (1.0, 1.0, 300.0, 30.0),
    r_force: float = 0.12,
) -> np.ndarray:
    a, b = _cartpole_linearized(mc, mp, half_length)
    ad, bd = _discretize_euler(a, b, dt)
    q = np.diag(q_diag)
    r = np.array([[r_force]], dtype=np.float64)
    return _solve_discrete_lqr(ad, bd, q, r)


class LQRBalanceController:
    """
    Continuous LQR with smooth pulse-width actuation for bang-bang motor.

    Uses light encoder scaling (balance-only) — not the full RL observation map.
  """

    def __init__(self) -> None:
        self.upright_angle = _env_float("CARTPOLE_UPRIGHT_ANGLE", 580.0)
        self.counts_per_rad = _env_float("CARTPOLE_COUNTS_PER_RAD", 400.0)
        self.meters_per_belt = _env_float("CARTPOLE_METERS_PER_BELT", 3.0e-5)
        self.force_mag = _env_float("CARTPOLE_FORCE_MAG", 10.0)

        self.min_pulse_ms = int(_env_float("CARTPOLE_BALANCE_MIN_PULSE_MS", 8))
        self.max_pulse_ms = int(_env_float("CARTPOLE_BALANCE_MAX_PULSE_MS", 55))
        self.deadband_force = _env_float("CARTPOLE_BALANCE_DEADBAND", 0.35)
        self.gain_scale = _env_float("CARTPOLE_LQR_GAIN_SCALE", 1.0)

        k = default_lqr_gain()
        k_override = os.environ.get("CARTPOLE_LQR_K")
        if k_override:
            k = np.array([float(x) for x in k_override.split(",")], dtype=np.float64).reshape(
                1, 4
            )
        self.k = k * self.gain_scale

        self._filt = np.zeros(4, dtype=np.float64)
        self._alpha = _env_float("CARTPOLE_BALANCE_FILTER", 0.35)

    def reset(self) -> None:
        self._filt[:] = 0.0

    def _to_state(self, obs: np.ndarray) -> np.ndarray:
        angle, belt, ang_vel, belt_vel = (float(v) for v in obs[:4])
        theta = (angle - self.upright_angle) / self.counts_per_rad
        x = belt * self.meters_per_belt
        theta_dot = ang_vel / self.counts_per_rad
        x_dot = belt_vel * self.meters_per_belt
        raw = np.array([x, x_dot, theta, theta_dot], dtype=np.float64)
        self._filt = self._alpha * raw + (1.0 - self._alpha) * self._filt
        return self._filt

    def act(self, obs: np.ndarray) -> BalanceCommand:
        state = self._to_state(obs)
        force = float(-(self.k @ state).item())
        force = float(np.clip(force, -self.force_mag, self.force_mag))

        if abs(force) < self.deadband_force:
            return BalanceCommand(action=2, pulse_ms=0, force=force)

        action = 1 if force > 0 else 0
        ratio = min(abs(force) / self.force_mag, 1.0)
        pulse_ms = int(
            self.min_pulse_ms + ratio * (self.max_pulse_ms - self.min_pulse_ms)
        )
        return BalanceCommand(action=action, pulse_ms=pulse_ms, force=force)
