import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
import os
import pandas as pd

# Import your environment
from train_cartpole_swingup import CartPoleSwingUpEnv

class CartPoleVisualization:
    """Comprehensive visualization and analysis for CartPole Swing-Up"""
    
    def __init__(self, model_path="cartpole_swingup_ppo", logs_dir="./logs/"):
        self.model_path = model_path
        self.logs_dir = logs_dir
        self.model = None
        
    def load_model(self):
        """Load the trained PPO model"""
        try:
            self.model = PPO.load(self.model_path)
            print(f"✅ Model loaded successfully from {self.model_path}.zip")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def plot_training_rewards(self):
        """Plot training progress from evaluation logs"""
        
        # Try to load evaluation results
        eval_file = os.path.join(self.logs_dir, "evaluations.npz")
        
        if os.path.exists(eval_file):
            try:
                data = np.load(eval_file)
                timesteps = data['timesteps']
                results = data['results']
                ep_lengths = data['ep_lengths']
                
                print(f"Loaded evaluation data: {len(timesteps)} evaluation points")
                print(f"Results shape: {np.array(results).shape}")
                
                # Ensure results is 1D array
                if len(np.array(results).shape) > 1:
                    results = np.mean(results, axis=1)  # Take mean if multiple episodes per eval
                
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                
                # Plot 1: Mean reward over time
                axes[0,0].plot(timesteps, results, linewidth=2, color='blue')
                axes[0,0].set_xlabel('Timesteps')
                axes[0,0].set_ylabel('Mean Episode Reward')
                axes[0,0].set_title('Training Progress: Mean Reward')
                axes[0,0].grid(True, alpha=0.3)
                
                # Plot 2: Episode lengths
                if len(np.array(ep_lengths).shape) > 1:
                    ep_lengths = np.mean(ep_lengths, axis=1)
                axes[0,1].plot(timesteps, ep_lengths, linewidth=2, color='green')
                axes[0,1].set_xlabel('Timesteps')
                axes[0,1].set_ylabel('Mean Episode Length')
                axes[0,1].set_title('Training Progress: Episode Length')
                axes[0,1].grid(True, alpha=0.3)
                
                # Plot 3: Reward distribution (recent episodes)
                recent_results = results[-10:] if len(results) > 10 else results
                # Flatten in case there are still nested arrays
                recent_results = np.array(recent_results).flatten()
                axes[1,0].hist(recent_results, bins=min(15, len(recent_results)), alpha=0.7, color='orange')
                axes[1,0].set_xlabel('Episode Reward')
                axes[1,0].set_ylabel('Frequency')
                axes[1,0].set_title('Recent Reward Distribution')
                axes[1,0].grid(True, alpha=0.3)
                
                # Plot 4: Rolling average
                window = max(3, len(results) // 10)
                rolling_mean = pd.Series(results).rolling(window=window, min_periods=1).mean()
                axes[1,1].plot(timesteps, results, alpha=0.3, color='blue', label='Raw')
                axes[1,1].plot(timesteps, rolling_mean, linewidth=2, color='red', label=f'Rolling Mean ({window})')
                axes[1,1].set_xlabel('Timesteps')
                axes[1,1].set_ylabel('Reward')
                axes[1,1].set_title('Training Progress with Smoothing')
                axes[1,1].legend()
                axes[1,1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.show()
                
                # Print summary statistics
                print("\n📊 Training Summary:")
                print(f"Final mean reward: {results[-1]:.2f}")
                print(f"Best reward achieved: {np.max(results):.2f}")
                print(f"Final episode length: {ep_lengths[-1]:.1f}")
                print(f"Total training timesteps: {timesteps[-1]:,}")
                
            except Exception as e:
                print(f"❌ Error loading evaluation data: {e}")
                print("Trying alternative log formats...")
                self._try_alternative_logs()
                
        else:
            print(f"❌ No evaluation log found at {eval_file}")
            print("Trying alternative log formats...")
            self._try_alternative_logs()
    
    def _try_alternative_logs(self):
        """Try to load alternative log formats"""
        
        # Try Monitor CSV logs
        monitor_file = os.path.join(self.logs_dir, "monitor.csv")
        if os.path.exists(monitor_file):
            try:
                print(f"📁 Found monitor.csv, plotting from there...")
                
                # Skip header lines that start with #
                with open(monitor_file, 'r') as f:
                    lines = f.readlines()
                
                # Find where data starts (after comment lines)
                data_start = 0
                for i, line in enumerate(lines):
                    if not line.startswith('#'):
                        data_start = i
                        break
                
                # Read the CSV data
                df = pd.read_csv(monitor_file, skiprows=data_start)
                
                if 'r' in df.columns and 'l' in df.columns and 't' in df.columns:
                    rewards = df['r'].values
                    lengths = df['l'].values
                    times = df['t'].values
                    
                    # Create plots
                    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                    
                    # Episodes vs rewards
                    episodes = range(len(rewards))
                    axes[0,0].plot(episodes, rewards, alpha=0.6, color='blue')
                    axes[0,0].set_xlabel('Episode')
                    axes[0,0].set_ylabel('Episode Reward')
                    axes[0,0].set_title('Episode Rewards Over Time')
                    axes[0,0].grid(True, alpha=0.3)
                    
                    # Episodes vs lengths
                    axes[0,1].plot(episodes, lengths, alpha=0.6, color='green')
                    axes[0,1].set_xlabel('Episode')
                    axes[0,1].set_ylabel('Episode Length')
                    axes[0,1].set_title('Episode Lengths Over Time')
                    axes[0,1].grid(True, alpha=0.3)
                    
                    # Reward distribution
                    axes[1,0].hist(rewards, bins=min(30, len(rewards)//5), alpha=0.7, color='orange')
                    axes[1,0].set_xlabel('Episode Reward')
                    axes[1,0].set_ylabel('Frequency')
                    axes[1,0].set_title('Reward Distribution')
                    axes[1,0].grid(True, alpha=0.3)
                    
                    # Rolling averages
                    window = max(10, len(rewards) // 20)
                    rolling_rewards = pd.Series(rewards).rolling(window=window, min_periods=1).mean()
                    axes[1,1].plot(episodes, rewards, alpha=0.3, color='blue', label='Raw')
                    axes[1,1].plot(episodes, rolling_rewards, linewidth=2, color='red', label=f'Rolling Mean ({window})')
                    axes[1,1].set_xlabel('Episode')
                    axes[1,1].set_ylabel('Reward')
                    axes[1,1].set_title('Training Progress')
                    axes[1,1].legend()
                    axes[1,1].grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    plt.show()
                    
                    print(f"\n📊 Training Summary (from {len(rewards)} episodes):")
                    print(f"Final episode reward: {rewards[-1]:.2f}")
                    print(f"Best reward achieved: {np.max(rewards):.2f}")
                    print(f"Average reward (last 100 episodes): {np.mean(rewards[-100:]):.2f}")
                    print(f"Final episode length: {lengths[-1]:.0f}")
                    
                else:
                    print(f"❌ Monitor CSV doesn't have expected columns: {df.columns.tolist()}")
                    
            except Exception as e:
                print(f"❌ Error reading monitor.csv: {e}")
        
        else:
            print("❌ No training logs found!")
            print("Make sure you:")
            print("1. Trained with EvalCallback: callback=EvalCallback(...)")
            print("2. Used Monitor wrapper: env = Monitor(env)")
            print("3. Specified correct log_path in training")
    
    def visualize_single_episode(self, env, episode_length=1000, render_mode="plots"):
        """Run and visualize a single episode"""
        
        if self.model is None:
            print("❌ Model not loaded! Call load_model() first.")
            return
        
        obs, _ = env.reset()
        
        # Storage for trajectory data
        trajectory = {
            'timestep': [],
            'cart_pos': [],
            'cart_vel': [],
            'pole_angle': [],
            'pole_vel': [],
            'action': [],
            'reward': [],
            'cumulative_reward': []
        }
        
        cumulative_reward = 0
        
        print(f"🎬 Running episode visualization...")
        
        for step in range(episode_length):
            # Get action from trained model
            action, _ = self.model.predict(obs, deterministic=True)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            cumulative_reward += reward
            
            # Store trajectory data
            trajectory['timestep'].append(step)
            trajectory['cart_pos'].append(obs[0])
            trajectory['cart_vel'].append(obs[1])
            trajectory['pole_angle'].append(np.rad2deg(obs[2]))  # Convert to degrees
            trajectory['pole_vel'].append(obs[3])
            trajectory['action'].append(action)
            trajectory['reward'].append(reward)
            trajectory['cumulative_reward'].append(cumulative_reward)
            
            # Print progress occasionally
            if step % 100 == 0:
                angle_deg = np.rad2deg(obs[2])
                print(f"Step {step:3d}: Angle = {angle_deg:6.1f}°, Reward = {reward:6.3f}, Cum. Reward = {cumulative_reward:8.2f}")
            
            if terminated or truncated:
                print(f"Episode ended at step {step}")
                break
        
        # Plot trajectory
        if render_mode == "plots":
            self._plot_trajectory(trajectory)
        
        return trajectory
    
    def _plot_trajectory(self, trajectory):
        """Plot detailed trajectory analysis"""
        
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        
        timesteps = trajectory['timestep']
        
        # Cart position and velocity
        axes[0,0].plot(timesteps, trajectory['cart_pos'], color='blue', linewidth=2)
        axes[0,0].set_title('Cart Position', fontsize=14)
        axes[0,0].set_ylabel('Position (m)')
        axes[0,0].grid(True, alpha=0.3)
        
        axes[0,1].plot(timesteps, trajectory['cart_vel'], color='green', linewidth=2)
        axes[0,1].set_title('Cart Velocity', fontsize=14)
        axes[0,1].set_ylabel('Velocity (m/s)')
        axes[0,1].grid(True, alpha=0.3)
        
        # Pole angle and angular velocity
        axes[1,0].plot(timesteps, trajectory['pole_angle'], color='red', linewidth=2)
        axes[1,0].axhline(y=0, color='black', linestyle='--', alpha=0.5, label='Upright')
        axes[1,0].axhline(y=180, color='gray', linestyle='--', alpha=0.5, label='Hanging Down')
        axes[1,0].axhline(y=-180, color='gray', linestyle='--', alpha=0.5)
        axes[1,0].fill_between(timesteps, -12, 12, alpha=0.2, color='green', label='Balance Zone')
        axes[1,0].set_title('Pole Angle', fontsize=14)
        axes[1,0].set_ylabel('Angle (degrees)')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        axes[1,1].plot(timesteps, trajectory['pole_vel'], color='orange', linewidth=2)
        axes[1,1].set_title('Pole Angular Velocity', fontsize=14)
        axes[1,1].set_ylabel('Angular Velocity (rad/s)')
        axes[1,1].grid(True, alpha=0.3)
        
        # Actions and rewards
        axes[2,0].step(timesteps, trajectory['action'], where='mid', color='purple', linewidth=2)
        axes[2,0].set_title('Actions (0=Left, 1=Right)', fontsize=14)
        axes[2,0].set_ylabel('Action')
        axes[2,0].set_xlabel('Timestep')
        axes[2,0].set_ylim(-0.1, 1.1)
        axes[2,0].grid(True, alpha=0.3)
        
        axes[2,1].plot(timesteps, trajectory['cumulative_reward'], color='darkblue', linewidth=2)
        axes[2,1].set_title('Cumulative Reward', fontsize=14)
        axes[2,1].set_ylabel('Cumulative Reward')
        axes[2,1].set_xlabel('Timestep')
        axes[2,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Performance analysis
        final_angle = trajectory['pole_angle'][-1]
        time_upright = sum(1 for angle in trajectory['pole_angle'] if abs(angle) < 12)
        swing_up_time = None
        
        # Find when pole first reaches upright position
        for i, angle in enumerate(trajectory['pole_angle']):
            if abs(angle) < 12:
                swing_up_time = i
                break
        
        print(f"\n📈 Episode Analysis:")
        print(f"Final angle: {final_angle:.1f}°")
        print(f"Time spent upright (±12°): {time_upright}/{len(timesteps)} steps ({100*time_upright/len(timesteps):.1f}%)")
        if swing_up_time is not None:
            print(f"Swing-up time: {swing_up_time} steps")
        print(f"Final cumulative reward: {trajectory['cumulative_reward'][-1]:.2f}")
    
    def compare_multiple_episodes(self, env, num_episodes=5):
        """Compare performance across multiple episodes"""
        
        if self.model is None:
            print("❌ Model not loaded! Call load_model() first.")
            return
        
        episode_data = []
        
        print(f"🔄 Running {num_episodes} episodes for comparison...")
        
        for episode in range(num_episodes):
            print(f"\nEpisode {episode + 1}/{num_episodes}")
            
            # Run episode without plotting
            obs, _ = env.reset()
            trajectory = {
                'timestep': [],
                'pole_angle': [],
                'cumulative_reward': []
            }
            
            cumulative_reward = 0
            
            for step in range(1000):  # Max episode length
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                cumulative_reward += reward
                
                trajectory['timestep'].append(step)
                trajectory['pole_angle'].append(np.rad2deg(obs[2]))
                trajectory['cumulative_reward'].append(cumulative_reward)
                
                if terminated or truncated:
                    break
            # Calculate episode metrics
            final_angle = trajectory['pole_angle'][-1]
            time_upright = sum(1 for angle in trajectory['pole_angle'] if abs(angle) < 12)
            total_reward = trajectory['cumulative_reward'][-1]
            episode_length = len(trajectory['timestep'])
            
            episode_data.append({
                'episode': episode + 1,
                'final_angle': final_angle,
                'time_upright': time_upright,
                'upright_percentage': 100 * time_upright / episode_length,
                'total_reward': total_reward,
                'episode_length': episode_length
            })
            
            print(f"  Final angle: {final_angle:.1f}°, Total reward: {total_reward:.2f}")
        
        # Create comparison plots
        df = pd.DataFrame(episode_data)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Episode rewards
        axes[0,0].bar(df['episode'], df['total_reward'], color='skyblue')
        axes[0,0].set_title('Total Reward per Episode')
        axes[0,0].set_xlabel('Episode')
        axes[0,0].set_ylabel('Total Reward')
        
        # Time spent upright
        axes[0,1].bar(df['episode'], df['upright_percentage'], color='lightgreen')
        axes[0,1].set_title('Time Spent Upright (%)')
        axes[0,1].set_xlabel('Episode')
        axes[0,1].set_ylabel('Percentage')
        
        # Final angles
        axes[1,0].bar(df['episode'], df['final_angle'], color='salmon')
        axes[1,0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1,0].set_title('Final Pole Angle')
        axes[1,0].set_xlabel('Episode')
        axes[1,0].set_ylabel('Angle (degrees)')
        
        # Episode lengths
        axes[1,1].bar(df['episode'], df['episode_length'], color='gold')
        axes[1,1].set_title('Episode Length')
        axes[1,1].set_xlabel('Episode')
        axes[1,1].set_ylabel('Steps')
        
        plt.tight_layout()
        plt.show()
        
        # Print summary statistics
        print(f"\n📊 Multi-Episode Summary:")
        print(f"Average reward: {df['total_reward'].mean():.2f} ± {df['total_reward'].std():.2f}")
        print(f"Average upright time: {df['upright_percentage'].mean():.1f}% ± {df['upright_percentage'].std():.1f}%")
        print(f"Best episode reward: {df['total_reward'].max():.2f}")
        print(f"Most consistent episode: {df.loc[df['upright_percentage'].idxmax(), 'episode']}")

# Example usage
def main_visualization():
    """Main function to run all visualizations"""
    
    # Initialize visualizer
    viz = CartPoleVisualization(
        model_path="cartpole_swingup_ppo",  # Your saved model
        logs_dir="./logs/"
    )
    
    # Load the trained model
    if not viz.load_model():
        print("Cannot proceed without trained model!")
        return
    
    # Plot training progress
    print("📈 Plotting training progress...")
    viz.plot_training_rewards()
    
    # Create environment
    print("\n🎯 Creating environment and running visualizations...")
    env = CartPoleSwingUpEnv(render_mode="human")
    
    # Run single episode visualization
    print("\n🎬 Running single episode visualization...")
    trajectory = viz.visualize_single_episode(env)
    
    # Compare multiple episodes
    print("\n🔄 Running multi-episode comparison...")
    viz.compare_multiple_episodes(env, num_episodes=3)
    
    print("\n✅ Visualization complete!")

if __name__ == "__main__":
    # Run the full visualization suite
    main_visualization()
    
    # Or run individual components:
    # quick_plot_training()  # Just plot training progress
    # quick_test_model()     # Quick model test without plots

# Simple usage examples:
"""
# To use this script:

1. Make sure you have your trained model: cartpole_swingup_ppo.zip
2. Make sure you have training logs: ./logs/evaluations.npz
3. Run this script directly, or import and use:

from cartpole_visualization import CartPoleVisualization, quick_plot_training

# Quick training plot
quick_plot_training()

# Full analysis
viz = CartPoleVisualization()
viz.load_model()
env = CartPoleSwingUpEnv()
viz.visualize_single_episode(env)
viz.compare_multiple_episodes(env, num_episodes=5)
"""

# Quick visualization functions you can call directly:

def quick_plot_training(logs_dir="./logs/"):
    """Quick function to just plot training progress"""
    viz = CartPoleVisualization(logs_dir=logs_dir)
    viz.plot_training_rewards()

def quick_test_model(model_path="cartpole_swingup_ppo"):
    """Quick function to load and test model performance"""
    viz = CartPoleVisualization(model_path=model_path)
    if viz.load_model():
        print("✅ Model loaded successfully!")
        
        # Create environment and run a quick test
        env = CartPoleSwingUpEnv()
        print("🎯 Running quick test episode...")
        
        obs, _ = env.reset()
        total_reward = 0
        
        for step in range(500):
            action, _ = viz.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if step % 100 == 0:
                angle = np.rad2deg(obs[2])
                print(f"Step {step}: Angle = {angle:.1f}°, Reward = {reward:.3f}")
            
            if terminated or truncated:
                break
        
        print(f"\n📊 Quick Test Results:")
        print(f"Episode length: {step + 1} steps")
        print(f"Total reward: {total_reward:.2f}")
        print(f"Final angle: {np.rad2deg(obs[2]):.1f}°")
        
    else:
        print("❌ Could not load model")
