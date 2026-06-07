"""Phased CartPole reward: aggressive swing-up, then upright jittery balance."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def angle_from_upright(theta: float) -> float:
    """0 at upright; pi at hanging (theta wrapped to (-pi, pi])."""
    return float(abs(((theta + math.pi) % (2.0 * math.pi)) - math.pi))


@dataclass(frozen=True)
class RewardConfig:
  # Swing: wide arcs, commit force, cart excursion
  w_sweep: float = 0.38
  w_excursion: float = 0.10
  w_xdot_swing: float = 0.055
  xdot_swing_ref: float = 2.2
  w_commit: float = 0.15
  hang_rad: float = 2.15
  w_slack: float = 0.04
  # Hang / time
  time_pressure: float = 0.0065
  hang_urgency: float = 0.028
  w_xdot_hang: float = 0.00025
  # Phase boundaries (rad from upright)
  balance_rad: float = math.radians(18.0)
  swing_rad: float = math.radians(28.0)
  # Balance: stay up, small cart motion, live pole micro-motion
  upright_bonus: float = 0.8
  w_x_balance: float = 0.18
  w_xdot_balance: float = 0.10
  xdot_balance_ref: float = 0.85
  w_theta_dot_jitter: float = 0.07
  theta_dot_jitter_ref: float = 1.8


DEFAULT_REWARD_CONFIG = RewardConfig()

# Less cart-speed incentive — better match for real motor / encoder x_dot scale
SIM2REAL_REWARD_CONFIG = RewardConfig(
    w_excursion=0.05,
    w_xdot_swing=0.025,
    xdot_swing_ref=1.0,
    w_commit=0.10,
    upright_bonus=0.9,
    w_x_balance=0.20,
    w_xdot_balance=0.12,
)


def phased_cartpole_reward(
    x: float,
    x_dot: float,
    theta: float,
    theta_dot: float,
    action_u: float,
    *,
    x_threshold: float,
    cfg: RewardConfig = DEFAULT_REWARD_CONFIG,
) -> float:
    """action_u in [-1, 1] (discrete maps to +/-1)."""
    costheta = math.cos(theta)
    sintheta = math.sin(theta)
    ang = angle_from_upright(theta)
    au = float(np.clip(abs(action_u), 0.0, 1.0))

    r = costheta + 1.0 + cfg.w_sweep * (sintheta**2)

    if ang < cfg.balance_rad:
        r += cfg.upright_bonus
        r -= cfg.w_x_balance * (x / x_threshold) ** 2
        xdn = min(abs(x_dot) / cfg.xdot_balance_ref, 2.0)
        r -= cfg.w_xdot_balance * (xdn**2)
        td = min(abs(theta_dot), cfg.theta_dot_jitter_ref)
        r += cfg.w_theta_dot_jitter * (td / cfg.theta_dot_jitter_ref)
        r -= cfg.time_pressure * 0.2
    elif ang > cfg.swing_rad:
        r -= cfg.time_pressure
        if ang > math.pi / 2:
            r -= cfg.hang_urgency
            r -= cfg.w_xdot_hang * abs(x_dot)
        r += cfg.w_excursion * min(abs(x) / x_threshold, 1.0)
        if cfg.swing_rad < ang < math.pi - 0.35:
            r += cfg.w_xdot_swing * min(abs(x_dot) / cfg.xdot_swing_ref, 1.6)
        if ang > cfg.hang_rad:
            r += cfg.w_commit * au
            if abs(x_dot) < 0.35:
                r -= cfg.w_slack
    else:
        r -= cfg.time_pressure * 0.55

    return float(r)


def action_to_unit(action) -> float:
    a = int(np.asarray(action).item())
    return 1.0 if a == 1 else -1.0
