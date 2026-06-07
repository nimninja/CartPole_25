import os
import time

import gymnasium as gym
import numpy as np
import serial
from gymnasium import spaces

from real.encoder_state import (
    BELT_PER_SIM_X,
    angle_count_to_theta,
    physics_observation_from_state,
)
from cartpole_constants import MAX_EPISODE_STEPS, MOTOR_PULSE_S, STEP_DT
from sim.reward_shaping import SIM2REAL_REWARD_CONFIG, action_to_unit, phased_cartpole_reward

__all__ = ["CartPoleEnv", "MAX_EPISODE_STEPS", "STEP_DT"]


class CartPoleEnv(gym.Env):
    """Real CartPole: encoder counts in, sim-style obs + phased reward out."""

    def __init__(
        self,
        port=None,
        baudrate=9600,
        step_dt: float = STEP_DT,
        verbose: bool = False,
        max_episode_steps: int = MAX_EPISODE_STEPS,
    ):
        super().__init__()
        self.step_dt = step_dt
        self.verbose = verbose
        self.max_episode_steps = max_episode_steps
        self.current_step = 0
        self.reward_cfg = SIM2REAL_REWARD_CONFIG

        self.prev_angle = 0
        self.prev_belt = 0
        self.prev_theta = float(np.pi)
        self.prev_x = 0.0
        self.prev_time = time.time()

        if port is None:
            port = os.environ.get("CARTPOLE_SERIAL_PORT", "COM9")
        baud_env = os.environ.get("CARTPOLE_SERIAL_BAUD")
        if baud_env is not None:
            baudrate = int(baud_env)

        self.arduino = serial.Serial(port=port, baudrate=baudrate, timeout=1)
        self.arduino.reset_input_buffer()
        self.arduino.flush()
        time.sleep(2)

        self.action_space = spaces.Discrete(2)

        _theta_lim = float(np.pi)
        _x_lim = 2.4
        _high = np.array([_x_lim * 2, np.inf, _theta_lim, np.inf], dtype=np.float32)
        self.observation_space = spaces.Box(low=-_high, high=_high, dtype=np.float32)
        self.x_threshold = _x_lim

    def _log(self, *args) -> None:
        if self.verbose:
            print(*args)

    def _send_cmd(self, cmd: str) -> None:
        self.arduino.write(f"{cmd}\n".encode())
        self.arduino.flush()

    def _stop_motor(self) -> None:
        self._send_cmd("STOP")

    def _drain_serial(self, max_lines: int = 500) -> None:
        """Drop buffered telemetry so the next read is fresh."""
        old_timeout = self.arduino.timeout
        self.arduino.timeout = 0
        try:
            for _ in range(max_lines):
                if not self.arduino.readline():
                    break
        finally:
            self.arduino.timeout = old_timeout

    def pyserial_values(self) -> tuple[int, int]:
        while True:
            raw = self.arduino.readline().decode(errors="ignore").strip()
            if not raw or "," not in raw:
                continue
            self._log("RAW:", raw)
            try:
                angle_str, belt_str = raw.split(",")
                return int(angle_str), int(belt_str)
            except ValueError:
                self._log("Found formatting error")
                continue

    def _read_obs(self) -> np.ndarray:
        angle, belt = self.pyserial_values()
        self._log(f"angle: {angle}, belt: {belt}")
        obs, theta, x, _ = physics_observation_from_state(
            angle,
            belt,
            self.prev_angle,
            self.prev_belt,
            self.prev_theta,
            self.prev_x,
            self.step_dt,
        )
        self.prev_angle = angle
        self.prev_belt = belt
        self.prev_theta = theta
        self.prev_x = x
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    def step(self, action):
        self.current_step += 1

        if action == 0:
            self._send_cmd("LEFT")
        elif action == 1:
            self._send_cmd("RIGHT")
        elif action == 2:
            self._stop_motor()
        else:
            raise ValueError(f"invalid action {action!r}")

        # Default: motor stays on until next step (matches old behavior).
        # Optional pulse via CARTPOLE_MOTOR_PULSE — causes visible stutter if too short.
        pulse_s = float(os.environ.get("CARTPOLE_MOTOR_PULSE", str(MOTOR_PULSE_S)))
        pulse_s = min(max(pulse_s, 0.0), self.step_dt)
        if action in (0, 1) and pulse_s > 0:
            time.sleep(pulse_s)
            self._stop_motor()
            time.sleep(max(0.0, self.step_dt - pulse_s))
        else:
            time.sleep(self.step_dt)

        obs = self._read_obs()
        x, x_dot, theta, theta_dot = obs

        terminated = bool(abs(self.prev_belt) > 14000)
        truncated = bool(self.current_step >= self.max_episode_steps)

        if terminated:
            reward = -1.0
        else:
            reward = phased_cartpole_reward(
                float(x),
                float(x_dot),
                float(theta),
                float(theta_dot),
                action_to_unit(action),
                x_threshold=self.x_threshold,
                cfg=self.reward_cfg,
            )

        self._log(obs, reward, terminated, truncated)
        return obs, float(reward), terminated, truncated, {}

    def reset(self, seed=None, options=None):
        self._log("INTP RESET")
        self.current_step = 0

        # Stop cart and clear stale lines before RESET (pulse mode can leave motor mid-command)
        self._stop_motor()
        time.sleep(0.05)
        self._drain_serial()

        self._send_cmd("RESET")

        # Arduino homes belt in ~1–3 s; then wait for pole to be re-hung manually
        homing_wait = float(os.environ.get("CARTPOLE_RESET_HOMING_WAIT", "4"))
        time.sleep(homing_wait)
        self._drain_serial()

        reset_wait = float(os.environ.get("CARTPOLE_RESET_WAIT", "40"))
        if reset_wait > 0:
            time.sleep(reset_wait)
        self._drain_serial()

        angle, belt = self.pyserial_values()
        self.prev_angle = angle
        self.prev_belt = belt
        self.prev_x = float(belt) / BELT_PER_SIM_X
        self.prev_theta = angle_count_to_theta(angle)
        self.prev_time = time.time()
        time.sleep(self.step_dt)

        obs = self._read_obs()
        self._log("reset obs", obs)
        return obs, {}

    def close(self) -> None:
        if hasattr(self, "arduino") and self.arduino.is_open:
            self.arduino.close()
