import gymnasium as gym
from realenv import CartPoleEnv  # Ensure this matches your actual file and class name
from stable_baselines3 import DQN, PPO
from virtualenv import RealisticCartPoleEnv

env = CartPoleEnv()

# Load the trained model
model = PPO.load("cartpole_ppo_down")

obs, info = env.reset()

while True:
    action, _states = model.predict(obs, deterministic=True)

    obs, reward, terminated, truncated, info = env.step(action)

    print(f"Terminated: {terminated}, Truncated: {truncated}")

    if terminated or truncated:
        print(f"Resetting environment...")
        obs, info = env.reset()
