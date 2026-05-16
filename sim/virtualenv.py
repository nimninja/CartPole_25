import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.classic_control.cartpole import CartPoleEnv


class RealisticCartPoleEnv(CartPoleEnv):
    def __init__(self,
                 track_length=0.55,        # Half-length of track in meters
                 pole_length=0.35,        # Half-length of pole in meters
                 cart_mass=0.15,           # Mass of the cart in kg
                 pole_mass=0.06,          # Mass of the pole in kg
                 force_magnitude=5.0,     # Force applied by motor in Newtons
                 time_step=0.01,          # Time per simulation step (50Hz)
                 gravity=9.81,
                 render_mode = "human"):
        super().__init__(render_mode=render_mode)

        # Override physical parameters
        self.gravity = gravity
        self.masscart = cart_mass
        self.masspole = pole_mass
        self.total_mass = self.masspole + self.masscart
        self.length = pole_length  # Half of the pole length
        self.polemass_length = self.masspole * self.length
        self.force_mag = force_magnitude
        self.tau = time_step

        # Set thresholds
        self.theta_threshold_radians = 12 * 2 * np.pi / 360
        self.x_threshold = track_length

        # Adjust observation space
        high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float32).max,
            self.theta_threshold_radians * 2,
            np.finfo(np.float32).max
        ], dtype=np.float32)

        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

    def step(self, action):
        return super().step(action)

    def reset(self, seed=None, options=None):
        return super().reset(seed=seed, options=options)
