"""
PPO swing-up + raw-encoder balance.

Balance physics (cart catches the pole):
  lean = upright - angle          (+ = pole tipped low-angle side)
  To catch: accelerate cart so lean shrinks.
    lean > 0 and falling → push LEFT  (negative u)
    lean < 0 and falling → push RIGHT (positive u)
  Near upright: damp angular velocity, short pulses, release fast if lost.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
from stable_baselines3 import PPO


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(raw) if raw is not None else default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw is not None else default


@dataclass
class ControlOutput:
    action: int
    phase: str
    pulse_ms: int | None = None
    pid_u: float = 0.0
    err_angle: float = 0.0
    theta_deg: float = 0.0
    zone: str = ""
    rail: str = ""


class RailGuard:
    """Block cart motion into either motor / end stop."""

    def __init__(self) -> None:
        self.soft = _env_int("CARTPOLE_BELT_SOFT", 6200)
        self.hard = _env_int("CARTPOLE_BELT_HARD", 7600)

    def clamp(
        self, belt: float, action: int, pulse_ms: int | None = None
    ) -> tuple[int, int | None, str]:
        if action == 2:
            return action, pulse_ms, ""

        into_right = action == 1
        into_left = action == 0

        if belt >= self.hard and into_right:
            return 2, 0, "HARD_R"
        if belt <= -self.hard and into_left:
            return 2, 0, "HARD_L"
        if belt >= self.soft and into_right:
            return 2, 0, "SOFT_R"
        if belt <= -self.soft and into_left:
            return 2, 0, "SOFT_L"

        if pulse_ms is not None and pulse_ms > 0:
            if into_right and belt > 0:
                ratio = 1.0 - min(belt / self.soft, 1.0)
                pulse_ms = max(4, int(pulse_ms * max(ratio, 0.08)))
            elif into_left and belt < 0:
                ratio = 1.0 - min(abs(belt) / self.soft, 1.0)
                pulse_ms = max(4, int(pulse_ms * max(ratio, 0.08)))

        return action, pulse_ms, ""


class BalanceController:
    """
    Catch the pole: move cart to shrink lean, damp spin near top.

    Positive u → RIGHT.  Flip CARTPOLE_PID_SIGN if hardware is reversed.
    """

    def __init__(self, nominal_upright: int = 580, rail: RailGuard | None = None) -> None:
        self.nominal_upright = nominal_upright
        self._cpd = nominal_upright / 180.0

        self._kp = _env_float("CARTPOLE_PID_KP_ANGLE", 0.28)
        self._kd = _env_float("CARTPOLE_PID_KD_ANGLE", 0.028)
        self._kf = _env_float("CARTPOLE_PID_KF_FALL", 0.055)
        self._kx = _env_float("CARTPOLE_PID_KP_BELT", 0.0025)
        self._rail = rail if rail is not None else RailGuard()
        self._motor_sign = _env_float("CARTPOLE_PID_SIGN", 1.0)

        self._u_clip = _env_float("CARTPOLE_PID_U_CLIP", 90.0)
        self._deadband = _env_float("CARTPOLE_PID_DEADBAND", 0.05)
        self._min_pulse = int(_env_float("CARTPOLE_BALANCE_MIN_PULSE_MS", 4))
        self._max_pulse = int(_env_float("CARTPOLE_BALANCE_MAX_PULSE_MS", 16))
        self._full_motor_u = _env_float("CARTPOLE_PID_FULL_U", 5.0)

        self._vel_filt = 0.0
        self._vel_alpha = _env_float("CARTPOLE_PID_VEL_FILTER", 0.95)
        self._last_action = 0

    def reset(self, obs: np.ndarray | None = None) -> None:
        if obs is not None:
            self._vel_filt = float(obs[2])
        else:
            self._vel_filt = 0.0
        self._last_action = 0

    def _zone(self, lean_deg: float) -> str:
        if lean_deg >= 12.0:
            return "catch"
        if lean_deg >= 5.0:
            return "mid"
        return "fine"

    def _pulse_for_zone(self, zone: str, abs_u: float) -> int:
        caps = {"catch": self._max_pulse, "mid": 14, "fine": 8}
        cap = caps[zone]
        if abs_u >= self._full_motor_u:
            return cap
        ratio = min(abs_u / self._full_motor_u, 1.0)
        return max(self._min_pulse, int(self._min_pulse + ratio * (cap - self._min_pulse)))

    def act(self, obs: np.ndarray) -> ControlOutput:
        angle = float(obs[0])
        belt = float(obs[1])
        ang_vel = float(obs[2])

        self._vel_filt = self._vel_alpha * ang_vel + (1.0 - self._vel_alpha) * self._vel_filt

        lean = self.nominal_upright - angle
        theta_deg = lean / self._cpd
        lean_deg = abs(theta_deg)
        zone = self._zone(lean_deg)

        falling_away = (lean > 0 and self._vel_filt < 0) or (lean < 0 and self._vel_filt > 0)

        # P pulls lean toward 0; belt pulls cart toward center.
        if falling_away:
            # Catch the fall — drive cart under the pole (opposite to lean direction).
            catch = self._kp * abs(lean) + self._kf * abs(self._vel_filt)
            u = (-1.0 if lean > 0 else 1.0) * catch - 0.4 * self._kx * belt
        else:
            u = self._kp * lean - self._kd * self._vel_filt - self._kx * belt

        # Near top: max twitch — heavy D, full P.
        if lean_deg < 10.0 and not falling_away:
            u = 1.1 * self._kp * lean - 3.5 * self._kd * self._vel_filt - 0.7 * self._kx * belt

        u = float(np.clip(self._motor_sign * u, -self._u_clip, self._u_clip))

        # Always fire a pulse every cycle — never coast during balance.
        if abs(u) < self._deadband:
            u = self._deadband if (lean >= 0 or self._vel_filt >= 0) else -self._deadband
        action = 1 if u > 0 else 0
        pulse_ms = self._pulse_for_zone(zone, abs(u))

        action, pulse_ms, rail_note = self._rail.clamp(belt, action, pulse_ms)

        self._last_action = action if action != 2 else 0
        return ControlOutput(
            action=action,
            phase="balance",
            pulse_ms=pulse_ms,
            pid_u=u,
            err_angle=lean,
            theta_deg=theta_deg,
            zone=zone,
            rail=rail_note,
        )


class HybridController:
    def __init__(self) -> None:
        self.nominal_upright = _env_int("CARTPOLE_UPRIGHT_ANGLE", 580)
        self._latch_deg = _env_float("CARTPOLE_BALANCE_DEG", 48.0)
        self._release_deg = _env_float("CARTPOLE_RELEASE_DEG", 58.0)
        self.hang_down = _env_float("CARTPOLE_HANG_DOWN", 250.0)
        self._latch_vel = _env_float("CARTPOLE_LATCH_VEL", 950.0)
        self._latch_vel_fast = _env_float("CARTPOLE_LATCH_VEL_FAST", 1400.0)
        self._release_vel = _env_float("CARTPOLE_RELEASE_VEL", 520.0)
        self._fall_steps = 0
        self._fall_release = int(_env_float("CARTPOLE_FALL_RELEASE_STEPS", 6))
        self._fall_release_min_deg = _env_float("CARTPOLE_FALL_RELEASE_DEG", 16.0)
        self._cpd = self.nominal_upright / 180.0
        self._slack = self._latch_deg * self._cpd
        self._release_slack = self._release_deg * self._cpd

        self._rail = RailGuard()
        self.balance = BalanceController(self.nominal_upright, self._rail)
        self._balance_active = False
        self.just_latched = False
        self.just_released = False
        self._last_phase = "swing"

    @property
    def latch_deg(self) -> float:
        return self._latch_deg

    @property
    def band_low(self) -> float:
        return self.nominal_upright - self._slack

    @property
    def band_high(self) -> float:
        return self.nominal_upright + self._slack

    @property
    def release_deg(self) -> float:
        return self._release_deg

    @property
    def release_low(self) -> float:
        return self.nominal_upright - self._release_slack

    @property
    def release_high(self) -> float:
        return self.nominal_upright + self._release_slack

    @property
    def rail(self) -> RailGuard:
        return self._rail

    @property
    def last_phase(self) -> str:
        return self._last_phase

    def degrees_from_upright(self, angle: float) -> float:
        return abs(angle - self.nominal_upright) / self._cpd

    def reset(self) -> None:
        self._balance_active = False
        self.just_latched = False
        self.just_released = False
        self._fall_steps = 0
        self._last_phase = "swing"
        self.balance.reset()

    def _is_falling_away(self, angle: float, ang_vel: float) -> bool:
        lean = self.nominal_upright - angle
        return (lean > 0 and ang_vel < 0) or (lean < 0 and ang_vel > 0)

    def _approaching_upright(self, angle: float, ang_vel: float) -> bool:
        lean = self.nominal_upright - angle
        if lean > 0:
            return ang_vel > 0
        if lean < 0:
            return ang_vel < 0
        return True

    def _should_latch(self, angle: float, ang_vel: float) -> bool:
        if angle <= self.hang_down:
            return False
        if abs(angle - self.nominal_upright) > self._slack:
            return False
        if abs(ang_vel) <= self._latch_vel:
            return True
        # Swing-up: latch while moving toward upright even if still fast.
        return (
            self._approaching_upright(angle, ang_vel)
            and abs(ang_vel) <= self._latch_vel_fast
        )

    def _should_release(self, angle: float, ang_vel: float) -> bool:
        dist = self.degrees_from_upright(angle)
        if angle <= self.hang_down:
            return True
        if dist > self._release_deg:
            return True
        if abs(ang_vel) > self._release_vel and dist > 22.0:
            return True
        if self._is_falling_away(angle, ang_vel):
            self._fall_steps += 1
        else:
            self._fall_steps = 0
        return self._fall_steps >= self._fall_release and dist > self._fall_release_min_deg

    def _update_phase(self, angle: float, obs: np.ndarray) -> None:
        self.just_latched = False
        self.just_released = False
        was = self._balance_active
        ang_vel = float(obs[2])

        if self._balance_active:
            if self._should_release(angle, ang_vel):
                self._balance_active = False
                self._fall_steps = 0
        elif self._should_latch(angle, ang_vel):
            self._balance_active = True

        if not was and self._balance_active:
            self.just_latched = True
            self.balance.reset(obs)
        elif was and not self._balance_active:
            self.just_released = True

    def act(self, obs: np.ndarray, model: PPO) -> ControlOutput:
        angle = float(obs[0])
        self._update_phase(angle, obs)

        if self._balance_active:
            self._last_phase = "balance"
            return self.balance.act(obs)

        self._last_phase = "swing"
        action, _ = model.predict(obs, deterministic=True)
        action = int(np.asarray(action).reshape(-1)[0])
        belt = float(obs[1])
        action, _, rail_note = self._rail.clamp(belt, action, None)
        return ControlOutput(
            action=action,
            phase="swing",
            pulse_ms=None,
            rail=rail_note,
        )
