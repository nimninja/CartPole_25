import os
import time

import gymnasium as gym
import numpy as np
import serial
from gymnasium import spaces

try:
    from sim.actions import FORCE_HIGH, FORCE_LOW, clip_force, force_to_serial_line
except ImportError:
    import sys
    from pathlib import Path

    _root = Path(__file__).resolve().parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
    from sim.actions import FORCE_HIGH, FORCE_LOW, clip_force, force_to_serial_line


class CartPoleEnv(gym.Env):
    """Real CartPole: continuous force command in [-1, 1] (direction × speed)."""

    def __init__(self, port=None, baudrate=9600, verbose: bool = True):
        super().__init__()
        self.verbose = verbose
        self.prev_angle = 0
        self.prev_belt = 0
        self.prev_time = time.time()
        self.current_step = 0

        if port is None:
            port = os.environ.get("CARTPOLE_SERIAL_PORT", "COM9")
        baud_env = os.environ.get("CARTPOLE_SERIAL_BAUD")
        if baud_env is not None:
            baudrate = int(baud_env)

        self.arduino = serial.Serial(port=port, baudrate=baudrate, timeout=1)
        self.arduino.reset_input_buffer()
        self.arduino.flush()
        time.sleep(2)

        self.action_space = spaces.Box(
            low=np.array([FORCE_LOW], dtype=np.float32),
            high=np.array([FORCE_HIGH], dtype=np.float32),
            dtype=np.float32,
        )

        _theta_lim = 12 * 2 * np.pi / 360
        _x_lim = 2.4
        _high = np.array(
            [_x_lim * 2, np.inf, _theta_lim * 2, np.inf],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=-_high, high=_high, dtype=np.float32)

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def pyserial_values(self):
        while True:
            raw = self.arduino.readline().decode(errors="ignore").strip()
            if not raw or "," not in raw:
                continue
            self._log(f"RAW: {raw}")
            try:
                angle_str, belt_str = raw.split(",", 1)
                angle = int(angle_str)
                belt = int(belt_str)
                break
            except ValueError:
                self._log("Found formatting error")
                continue

        self._log(f"angle: {angle}, belt: {belt}")
        now = time.time()
        delta = max(now - self.prev_time, 0.001)
        ang_vel = (angle - self.prev_angle) / delta
        belt_vel = (belt - self.prev_belt) / delta
        self.prev_time = now
        self.prev_angle = angle
        self.prev_belt = belt
        return np.array([angle, belt, ang_vel, belt_vel], dtype=np.float32)

    def step(self, action):
        self.current_step += 1
        self.pyserial_values()

        force = clip_force(action)
        self.arduino.write(force_to_serial_line(force))

        self.obs = self.pyserial_values()
        self.terminated = bool(self.obs[1] < -14000 or self.obs[1] > 14000)
        self.truncated = bool(self.current_step >= 1000)

        self.reward = 5.0
        a0 = self.obs[0]
        if 0 < a0 <= 200:
            self.reward += a0 * 0.03
        elif 200 < a0 <= 500:
            self.reward += a0 * 0.08
        elif 500 < a0 <= 700:
            self.reward += 60.0
        elif 700 < a0 <= 1000:
            self.reward += (1200 - a0) * 0.08
        elif 1000 <= a0 < 1200:
            self.reward += (1200 - a0) * 0.03
        if self.terminated:
            self.reward -= 500

        self.observation = self.obs
        self._log(
            f"force={force:.3f} obs={self.observation} "
            f"r={self.reward} term={self.terminated} trunc={self.truncated}"
        )
        return self.observation, float(self.reward), self.terminated, self.truncated, {}

    def reset(self, seed=None, options=None):
        self._log("RESET")
        self.current_step = 0
        self.arduino.flush()
        self.arduino.write(b"RESET\n")
        self.arduino.flush()
        time.sleep(4.0)
        self.arduino.reset_input_buffer()
        self.observation = self.pyserial_values()
        self.observation[0], self.observation[1] = 0, 0
        self.observation[2], self.observation[3] = 0, 0
        self.prev_angle = 0
        self.prev_belt = 0
        self.prev_time = time.time()
        self.arduino.write(force_to_serial_line(0.0))
        time.sleep(0.5)
        return self.observation, {}
