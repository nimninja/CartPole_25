from typing import Optional
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt

# Your custom CartPole environment (paste the environment code here or import it)
class CartPoleSwingUpEnv(gym.Env):
    """CartPole Swing-Up Environment"""
    
    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        
        # Physical parameters
        self.gravity = 9.81
        self.masscart = 0.15
        self.masspole = 0.06
        self.total_mass = self.masspole + self.masscart
        self.length = 0.265  # Half pole length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 1.0
        self.tau = 0.01  # Integration time step
        
        # Environment bounds (for observation space)
        self.x_threshold = 2.4
        self.theta_threshold = np.pi  # Full 360 degrees
        
        # Action space: 0 = left, 1 = right
        self.action_space = gym.spaces.Discrete(2)
        
        # Observation space: [x, x_dot, theta, theta_dot]
        high = np.array([
            self.x_threshold * 2,
            np.inf,
            np.inf,  # Allow full rotation
            np.inf,
        ], dtype=np.float32)
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)
        
        self.render_mode = render_mode
        self.state = None
        self.step_count = 0
        self.max_episode_steps = 1000  # Add episode length limit
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Start with pole hanging down (π radians from vertical)
        # Add small random perturbation
        self.state = np.array([
            np.random.uniform(-0.1, 0.1),  # Cart position
            np.random.uniform(-0.1, 0.1),  # Cart velocity
            np.pi + np.random.uniform(-0.1, 0.1),  # Pole angle (hanging down)
            np.random.uniform(-0.1, 0.1)   # Pole angular velocity
        ], dtype=np.float64)
        
        self.step_count = 0
        return np.array(self.state, dtype=np.float32), {}
    
    def step(self, action):
        assert self.action_space.contains(action)
        
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        
        # Physics simulation (from your environment)
        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        
        temp = (force + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0/3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        
        # Euler integration
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        
        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float64)
        self.step_count += 1
        
        # Reward function (enhanced version of yours)
        reward = self._compute_reward(x, x_dot, theta, theta_dot)
        
        # Termination conditions
        terminated = False  # No failure termination as requested
        
        # Add safety bounds to prevent cart from going too far
        if abs(x) > 3.0:  # Wider bounds than standard CartPole
            terminated = True
            reward = -10  # Penalty for going out of bounds
        
        # Truncation based on episode length
        truncated = self.step_count >= self.max_episode_steps
        
        obs = np.array(self.state, dtype=np.float32)
        info = {"step_count": self.step_count}
        
        return obs, reward, terminated, truncated, info
    
    def _compute_reward(self, x, x_dot, theta, theta_dot):
        """Enhanced reward function for swing-up task"""
        
        # Normalize angle to [-π, π]
        theta_normalized = ((theta + np.pi) % (2 * np.pi)) - np.pi
        
        # Main reward: cosine of angle (1 when upright, -1 when hanging)
        angle_reward = np.cos(theta_normalized)
        
        # Bonus for being upright (within ±12 degrees)
        upright_bonus = 0.0
        if abs(theta_normalized) < np.deg2rad(12):
            upright_bonus = 2.0
        
        # Small penalties to encourage efficiency
        velocity_penalty = -0.001 * (theta_dot**2 + x_dot**2)
        position_penalty = -0.001 * x**2
        
        total_reward = angle_reward + upright_bonus + velocity_penalty + position_penalty
        
        return total_reward
    
    def render(self):
        # Simplified render - you can use your full render method
        if self.render_mode == "human":
            print(f"Step {self.step_count}: x={self.state[0]:.3f}, θ={np.rad2deg(self.state[2]):.1f}°")

# Training setup
def train_cartpole_swingup(model_path="cartpole_swingup_ppo"):
    """Train PPO agent on CartPole swing-up task"""
    
    # Create environment
    env = CartPoleSwingUpEnv()
    env = Monitor(env)  # Monitor for logging
    
    # Create vectorized environment for better sample efficiency
    vec_env = make_vec_env(lambda: CartPoleSwingUpEnv(), n_envs=16)
    
    # PPO hyperparameters optimized for swing-up task
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=2048,         # Steps per rollout
        batch_size=64,
        n_epochs=10,
        gamma=0.99,           # Discount factor
        gae_lambda=0.95,      # GAE parameter
        clip_range=0.2,       # PPO clip range
        ent_coef=0.01,        # Entropy coefficient for exploration
        vf_coef=0.5,          # Value function coefficient
        max_grad_norm=0.5,    # Gradient clipping
        verbose=1,
        device="auto"
    )
    
    # Evaluation environment
    eval_env = Monitor(CartPoleSwingUpEnv())
    
    # Callback for evaluation and early stopping
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./logs/",
        log_path="./logs/",
        eval_freq=10_000,
        deterministic=True,
        render=False
    )
    
    # Train the agent
    print("Starting training...")
    total_timesteps = 2_000_000
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        progress_bar=True
    )
    
    # Save the final model
    model.save("cartpole_swingup_ppo")
    print("Training completed! Model saved as 'cartpole_swingup_ppo'")
    
    return model

def test_trained_model(model_path="cartpole_swingup_ppo"):
    """Test the trained model"""
    
    # Load the trained model
    model = PPO.load(model_path)
    
    # Create test environment
    env = CartPoleSwingUpEnv(render_mode="human")
    
    # Test for multiple episodes
    for episode in range(5):
        obs, _ = env.reset()
        total_reward = 0
        step_count = 0
        
        print(f"\n=== Episode {episode + 1} ===")
        
        for step in range(1000):  # Max steps per episode
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
            
            # Print progress every 100 steps
            if step % 100 == 0:
                theta_deg = np.rad2deg(obs[2])
                print(f"Step {step}: Angle = {theta_deg:.1f}°, Reward = {reward:.3f}")
            
            if terminated or truncated:
                break
        
        print(f"Episode {episode + 1} completed: {step_count} steps, Total reward: {total_reward:.2f}")
    
    env.close()

# Main training script
if __name__ == "__main__":
    # Uncomment to train
    trained_model = train_cartpole_swingup()
    
    # Uncomment to test trained model
    # test_trained_model()

# Additional utility functions
def plot_training_progress(log_path="./logs/evaluations.npz"):
    """Plot training progress"""
    try:
        data = np.load(log_path)
        timesteps = data['timesteps']
        results = data['results']
        
        plt.figure(figsize=(10, 6))
        plt.plot(timesteps, results)
        plt.xlabel('Timesteps')
        plt.ylabel('Mean Reward')
        plt.title('CartPole Swing-Up Training Progress')
        plt.grid(True)
        plt.savefig("training_progress.png")
        plt.close()
    except FileNotFoundError:
        print("No training log found. Train the model first.")

def analyze_solution_strategy(model_path="cartpole_swingup_ppo"):
    """Analyze how the agent solves the swing-up task"""
    model = PPO.load(model_path)
    env = CartPoleSwingUpEnv()
    
    obs, _ = env.reset()
    states = []
    actions = []
    rewards = []
    
    for step in range(500):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        states.append(obs.copy())
        actions.append(action)
        rewards.append(reward)
        
        if terminated or truncated:
            break
    
    states = np.array(states)
    
    # Plot the trajectory
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Cart position
    axes[0,0].plot(states[:, 0])
    axes[0,0].set_title('Cart Position')
    axes[0,0].set_ylabel('Position')
    
    # Pole angle
    axes[0,1].plot(np.rad2deg(states[:, 2]))
    axes[0,1].set_title('Pole Angle')
    axes[0,1].set_ylabel('Angle (degrees)')
    axes[0,1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    # Actions
    axes[1,0].plot(actions)
    axes[1,0].set_title('Actions (0=Left, 1=Right)')
    axes[1,0].set_ylabel('Action')
    axes[1,0].set_xlabel('Time Step')
    
    # Rewards
    axes[1,1].plot(rewards)
    axes[1,1].set_title('Rewards')
    axes[1,1].set_ylabel('Reward')
    axes[1,1].set_xlabel('Time Step')
    
    plt.tight_layout()
    plt.savefig("solution_strategy.png")
    plt.close()
    
    print(f"Final angle: {np.rad2deg(states[-1, 2]):.1f} degrees")
    print(f"Total reward: {sum(rewards):.2f}")
