from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from fullvirenv import CartPoleEnv  # Make sure the filename matches
import time

env = CartPoleEnv(render_mode=None)

# Wrap it in DummyVecEnv for SB3 compatibility
env = DummyVecEnv([lambda: env])

# Initialize the model

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="./cartpole_tensorboard/"
)

# Train the model
model.learn(total_timesteps=100_000)

# Save the model
model.save("cartpole_ppo_down")
#Load existing model for further training.
model = PPO.load("cartpole_ppo_down", env=env)

obs = env.reset()
done = False
while not done:
    action, _states = model.predict(obs)
    print(action)
    obs, reward, terminated, info = env.step(action)
    env.render()  # Render to visualize agent's actions
env.close()