import gymnasium as gym
import numpy as np
from gymnasium import spaces
import serial
import time


class CartPoleEnv(gym.Env):
    """Custom Environment for real-world CartPole setup."""

    #metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self):
        super().__init__()
        self.prev_angle = 0
        self.prev_belt = 0
        self.prev_time = time.time()
        self.current_step = 0

        # Serial setup
        self.arduino = serial.Serial(port='COM7', baudrate=9600, timeout=1)
        self.arduino.reset_input_buffer()
        self.arduino.flush()



        time.sleep(2)

        # Action: 0 = LEFT, 1 = RIGHT
        self.action_space = spaces.Discrete(2)

        # Observation: [angle, belt, angle_velocity, belt_velocity]
        obs_low = np.array([-np.inf, -np.inf, -np.inf, -np.inf], dtype=np.float32)
        obs_high = np.array([np.inf, np.inf, np.inf, np.inf], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

    def pyserial_values(self):

        while True:
            raw = self.arduino.readline().decode(errors='ignore').strip()
            if not raw or ',' not in raw:
                continue  # Skip empty or bad lines

            print("RAW:", raw)
            try:
                angle_str, belt_str = raw.split(",")
                angle = int(angle_str)
                belt = int(belt_str)
                break
            except ValueError:
                print("Found formatting error")
                continue

        print(f"angle: {angle}, belt: {belt}")
        # Calculate the velocity
        now = time.time()
        delta = now - self.prev_time if hasattr(self, "prev_time") else 0.05
        delta = max(delta, 0.001)

        ang_vel = (angle - self.prev_angle) / delta
        belt_vel = (belt - self.prev_belt) / delta

        self.prev_time = now
        self.prev_angle = angle
        self.prev_belt = belt

        return np.array([angle, belt, ang_vel, belt_vel], dtype=np.float32)

    def step(self, action):
        self.current_step += 1
        self.obs_before = self.pyserial_values()
        # Send the action to Arduino
        if action == 0:
            self.arduino.write(b"LEFT\n")
        elif action == 1:
            self.arduino.write(b"RIGHT\n")
        elif action == 2:
            self.arduino.write(b"STOP\n")
        # Give Arduino some time to move
        #time.sleep(0.08)  # Allow some time for Arduino to process the movement
        # Flush some bad lines right after opening serial

        # Get the updated encoder values

#        movement = 0
#        while movement < 100 :
#            time.sleep(0.01)
#            self.obs = self.pyserial_values()
#            movement = abs(self.obs_before[1] - self.obs[1])
#            print(f"This is in movement loop; {self.obs}")
        self.obs = self.pyserial_values()
        # Check if terminated or truncated
        self.terminated = bool(self.obs[1] < -15000 or self.obs[1] > 15000)
        self.truncated = bool(self.current_step >= 1000)

        # Calculate reward
        self.reward = 5.0
        if 0 < self.obs[0] <= 200:
            self.reward += self.obs[0] * 0.03
        elif 200 < self.obs[0] <= 500:
            self.reward += self.obs[0] * 0.08
        elif 500 < self.obs[0] <= 700:
            self.reward += 60.0
        elif 700 < self.obs[0] <= 1000:
            self.reward += (1200 - self.obs[0]) * 0.08
        elif 1000 <= self.obs[0] < 1200:
            self.reward += (1200 - self.obs[0]) * 0.03
        if self.terminated:
            self.reward -= 500
        self.info = {}
        self.observation = self.obs
        print(self.observation, self.reward, self.terminated, self.truncated)
        return self.observation, float(self.reward), self.terminated, self.truncated, self.info

    def reset(self, seed=None, options=None):
        print("INTP RESET")
        self.current_step = 0
        self.arduino.write(b"RESET\n")

        # Flush buffer
#        self.arduino.reset_input_buffer()

        # Wait for good reading
        '''
        while True:
            try:
                line = self.arduino.readline().decode(errors='ignore').strip()
                if ',' in line:
                    angle_str, belt_str = line.split(",")
                    angle = int(angle_str)
                    belt = int(belt_str)
                    break

            except:
                continue


        # Reset state tracking
        self.prev_angle = angle
        self.prev_belt = belt
        self.prev_time = time.time()
        time.sleep(5)
        '''
        self.arduino.flush()
        self.arduino.reset_input_buffer()
        self.observation = self.pyserial_values()
        self.observation[0], self.observation[1] = 0, 0
        time.sleep(3)
        return self.observation, {}

