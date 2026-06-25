import os

import time

import gymnasium as gym

import numpy as np

import serial

from gymnasium import spaces

try:
    from .balance import RailGuard
except ImportError:
    from balance import RailGuard





class CartPoleEnv(gym.Env):

    """Custom Environment for real-world CartPole setup."""



    #metadata = {"render_modes": ["human"], "render_fps": 30}



    def __init__(self, port=None, baudrate=9600, pulse_ms=25, verbose=False):

        super().__init__()

        self.prev_angle = 0

        self.prev_belt = 0

        self.prev_time = time.time()

        self.current_step = 0

        self.pulse_ms = int(os.environ.get("CARTPOLE_PULSE_MS", pulse_ms))
        self._min_balance_pulse = int(os.environ.get("CARTPOLE_BALANCE_MIN_PULSE_MS", 4))

        self.verbose = verbose or os.environ.get("CARTPOLE_VERBOSE", "0") == "1"
        self._rail = RailGuard()



        if port is None:

            port = os.environ.get("CARTPOLE_SERIAL_PORT", "COM3")

        baud_env = os.environ.get("CARTPOLE_SERIAL_BAUD")

        if baud_env is not None:

            baudrate = int(baud_env)



        self.arduino = serial.Serial(port=port, baudrate=baudrate, timeout=1)

        self.arduino.reset_input_buffer()

        self.arduino.flush()



        time.sleep(2)

        self._balance_loop_mode = False
        self._set_loop_mode(False)

        self.action_space = spaces.Discrete(2)



        # Must match sim/fullvirenv.CartPoleEnv observation_space (SB3 checks equality on PPO.load).

        # Same layout as training: [x, x_dot, theta, theta_dot] after your mapping.

        _theta_lim = 12 * 2 * np.pi / 360

        _x_lim = 2.4

        _high = np.array(

            [_x_lim * 2, np.inf, _theta_lim * 2, np.inf],

            dtype=np.float32,

        )

        self.observation_space = spaces.Box(low=-_high, high=_high, dtype=np.float32)



    def _log(self, *args) -> None:

        if self.verbose:

            print(*args)



    def set_loop_mode(self, balance: bool) -> None:
        self._set_loop_mode(balance)

    def _set_loop_mode(self, balance: bool) -> None:
        if balance == self._balance_loop_mode:
            return
        cmd = b"MODE BALANCE\n" if balance else b"MODE SWING\n"
        self.arduino.write(cmd)
        self.arduino.flush()
        time.sleep(0.05)
        self._balance_loop_mode = balance
        print(f"Arduino → {'BALANCE 3ms' if balance else 'SWING 60ms'}")

    def _send_motor(self, action: int) -> None:

        if action == 0:

            self.arduino.write(b"LEFT\n")

        elif action == 1:

            self.arduino.write(b"RIGHT\n")

        elif action == 2:

            self.arduino.write(b"STOP\n")

            return



        if self.pulse_ms > 0:

            time.sleep(self.pulse_ms / 1000.0)

            self.arduino.write(b"STOP\n")



    def pyserial_values(self):

        while True:

            raw = self.arduino.readline().decode(errors="ignore").strip()

            if not raw or "," not in raw:

                continue



            self._log("RAW:", raw)

            try:

                angle_str, belt_str = raw.split(",")

                angle = int(angle_str)

                belt = int(belt_str)

                break

            except ValueError:

                self._log("Found formatting error")

                continue



        self._log(f"angle: {angle}, belt: {belt}")

        now = time.time()

        delta = now - self.prev_time if hasattr(self, "prev_time") else 0.05

        delta = max(delta, 0.001)



        ang_vel = (angle - self.prev_angle) / delta

        belt_vel = (belt - self.prev_belt) / delta



        self.prev_time = now

        self.prev_angle = angle

        self.prev_belt = belt



        return np.array([angle, belt, ang_vel, belt_vel], dtype=np.float32)



    def _clamp_before_motor(
        self, action: int, pulse_ms: int | None
    ) -> tuple[int, int | None, str]:
        belt = float(self.prev_belt)
        return self._rail.clamp(belt, action, pulse_ms)

    def step(self, action, pulse_ms=None, balance=False):

        self.current_step += 1

        action = int(np.asarray(action).reshape(-1)[0])
        action, pulse_ms, rail_note = self._clamp_before_motor(action, pulse_ms)
        if rail_note and self.verbose:
            self._log(f"RAIL {rail_note} belt={self.prev_belt:.0f} → STOP")

        if balance:
            if action == 2:
                self.arduino.write(b"STOP\n")
            else:
                pulse = max(4, int(pulse_ms or self._min_balance_pulse))
                if action == 0:
                    self.arduino.write(b"LEFT\n")
                elif action == 1:
                    self.arduino.write(b"RIGHT\n")
                self.arduino.flush()
                time.sleep(pulse / 1000.0)
                self.arduino.write(b"STOP\n")
            self.arduino.flush()
            self.obs = self.pyserial_values()
        # Swing-up: original RL path (pulse_ms is None)
        elif pulse_ms is None:
            self.obs_before = self.pyserial_values()
            if action == 0:
                self.arduino.write(b"LEFT\n")
            elif action == 1:
                self.arduino.write(b"RIGHT\n")
            elif action == 2:
                self.arduino.write(b"STOP\n")
            self.obs = self.pyserial_values()
        else:
            if action == 2:
                self.arduino.write(b"STOP\n")
            else:
                saved_pulse = self.pulse_ms
                self.pulse_ms = int(pulse_ms)
                try:
                    self._send_motor(action)
                finally:
                    self.pulse_ms = saved_pulse
            self.obs = self.pyserial_values()



        self.terminated = bool(
            self.obs[1] < -self._rail.hard or self.obs[1] > self._rail.hard
        )

        self.truncated = bool(self.current_step >= 1000)



        self.reward = 5.0

        angle = self.obs[0]

        if 0 < angle <= 200:

            self.reward += angle * 0.03

        elif 200 < angle <= 500:

            self.reward += angle * 0.08

        elif 500 < angle <= 700:

            self.reward += 60.0

        elif 700 < angle <= 1000:

            self.reward += (1200 - angle) * 0.08

        elif 1000 <= angle < 1200:

            self.reward += (1200 - angle) * 0.03

        if self.terminated:

            self.reward -= 500



        self.info = {"action": action, "pulse_ms": self.pulse_ms}

        self.observation = self.obs

        self._log(self.observation, self.reward, self.terminated, self.truncated)

        return self.observation, float(self.reward), self.terminated, self.truncated, self.info



    def reset(self, seed=None, options=None):

        self._log("INTP RESET")

        self.current_step = 0

        self.arduino.flush()

        self.arduino.write(b"RESET\n")

        self.arduino.flush()

        time.sleep(40.0)

        self.arduino.reset_input_buffer()

        self._set_loop_mode(False)

        self.observation = self.pyserial_values()

        self.prev_angle = float(self.observation[0])
        self.prev_belt = float(self.observation[1])
        self.prev_time = time.time()



        time.sleep(0.5)

        return self.observation, {}



    def close(self):

        try:

            self.arduino.write(b"STOP\n")

            self.arduino.close()

        except Exception:

            pass

