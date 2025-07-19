#!/usr/bin/env python3
"""
Unified script to create demo videos and visualizations for CartPole swing-up
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Import the environment from the same directory
from train_cartpole_swingup import CartPoleSwingUpEnv
from stable_baselines3 import PPO

def create_analysis_plots(trajectory, output_dir="../videos"):
    """Create comprehensive analysis plots"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    fig = plt.figure(figsize=(15, 12))
    
    # 1. Cart and Pole positions at key moments
    key_frames = [0, len(trajectory['steps'])//4, len(trajectory['steps'])//2, 
                  3*len(trajectory['steps'])//4, len(trajectory['steps'])-1]
    
    ax1 = plt.subplot(3, 3, (1, 3))
    cart_width = 0.3
    pole_length = 0.5
    
    colors = ['red', 'orange', 'yellow', 'lightgreen', 'green']
    for i, frame in enumerate(key_frames):
        if frame < len(trajectory['steps']):
            x = trajectory['x'][frame]
            theta = trajectory['theta'][frame]
            
            # Cart
            cart_rect = plt.Rectangle((x - cart_width/2, -0.1), cart_width, 0.2, 
                                     fill=True, color=colors[i], alpha=0.7, 
                                     label=f'Step {trajectory["steps"][frame]}')
            ax1.add_patch(cart_rect)
            
            # Pole
            pole_x = x + pole_length * np.sin(theta)
            pole_y = pole_length * np.cos(theta)
            ax1.plot([x, pole_x], [0, pole_y], color=colors[i], linewidth=3, alpha=0.8)
            ax1.plot(pole_x, pole_y, 'o', color=colors[i], markersize=8)
    
    ax1.set_xlim(-3, 3)
    ax1.set_ylim(-1, 1)
    ax1.set_aspect('equal')
    ax1.set_title('CartPole Positions Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Pole angle over time
    ax2 = plt.subplot(3, 3, 4)
    angles_deg = [np.rad2deg(theta) for theta in trajectory['theta']]
    ax2.plot(trajectory['steps'], angles_deg, 'b-', linewidth=2)
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Upright')
    ax2.axhline(y=180, color='red', linestyle='--', alpha=0.5, label='Hanging')
    ax2.axhline(y=-180, color='red', linestyle='--', alpha=0.5)
    ax2.set_title('Pole Angle')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Angle (degrees)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Cart position over time
    ax3 = plt.subplot(3, 3, 5)
    ax3.plot(trajectory['steps'], trajectory['x'], 'g-', linewidth=2)
    ax3.set_title('Cart Position')
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Position')
    ax3.grid(True, alpha=0.3)
    
    # 4. Actions over time
    ax4 = plt.subplot(3, 3, 6)
    ax4.plot(trajectory['steps'], trajectory['actions'], 'r-', linewidth=1)
    ax4.set_title('Actions (0=Left, 1=Right)')
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Action')
    ax4.set_ylim(-0.1, 1.1)
    ax4.grid(True, alpha=0.3)
    
    # 5. Rewards over time
    ax5 = plt.subplot(3, 3, 7)
    ax5.plot(trajectory['steps'], trajectory['rewards'], 'purple', linewidth=1)
    ax5.set_title('Reward per Step')
    ax5.set_xlabel('Step')
    ax5.set_ylabel('Reward')
    ax5.grid(True, alpha=0.3)
    
    # 6. Cumulative reward
    ax6 = plt.subplot(3, 3, 8)
    cumulative_rewards = np.cumsum(trajectory['rewards'])
    ax6.plot(trajectory['steps'], cumulative_rewards, 'orange', linewidth=2)
    ax6.set_title('Cumulative Reward')
    ax6.set_xlabel('Step')
    ax6.set_ylabel('Total Reward')
    ax6.grid(True, alpha=0.3)
    
    # 7. Phase plot (angle vs angular velocity)
    ax7 = plt.subplot(3, 3, 9)
    ax7.plot(angles_deg, trajectory['theta_dot'], 'cyan', linewidth=1, alpha=0.7)
    ax7.scatter(angles_deg[0], trajectory['theta_dot'][0], color='red', s=100, label='Start')
    ax7.scatter(angles_deg[-1], trajectory['theta_dot'][-1], color='green', s=100, label='End')
    ax7.set_title('Phase Plot (Angle vs Angular Velocity)')
    ax7.set_xlabel('Angle (degrees)')
    ax7.set_ylabel('Angular Velocity (rad/s)')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    plt.tight_layout()
    analysis_path = os.path.join(output_dir, 'cartpole_analysis.png')
    plt.savefig(analysis_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Analysis plot saved: {analysis_path}")
    return analysis_path

def create_sequence_frames(trajectory, output_dir="../videos", num_frames=10):
    """Create a sequence of frames showing the motion"""
    
    frames_to_show = np.linspace(0, len(trajectory['steps'])-1, num_frames, dtype=int)
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    cart_width = 0.3
    cart_height = 0.15
    pole_length = 0.5
    
    for i, frame_idx in enumerate(frames_to_show):
        ax = axes[i]
        
        x = trajectory['x'][frame_idx]
        theta = trajectory['theta'][frame_idx]
        step = trajectory['steps'][frame_idx]
        angle_deg = np.rad2deg(theta)
        
        # Cart
        cart_rect = plt.Rectangle((x - cart_width/2, -cart_height/2), cart_width, cart_height, 
                                 fill=True, color='blue', alpha=0.8)
        ax.add_patch(cart_rect)
        
        # Pole
        pole_x = x + pole_length * np.sin(theta)
        pole_y = pole_length * np.cos(theta)
        ax.plot([x, pole_x], [0, pole_y], 'r-', linewidth=4)
        ax.plot(pole_x, pole_y, 'ro', markersize=10)
        
        # Ground
        ax.axhline(y=-cart_height/2, color='black', linewidth=2)
        
        # Formatting
        ax.set_xlim(-3, 3)
        ax.set_ylim(-1, 1)
        ax.set_aspect('equal')
        ax.set_title(f'Step {step}\nAngle: {angle_deg:.1f}°')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('CartPole Swing-Up Sequence', fontsize=16)
    plt.tight_layout()
    
    sequence_path = os.path.join(output_dir, 'cartpole_sequence.png')
    plt.savefig(sequence_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Sequence plot saved: {sequence_path}")
    return sequence_path

def create_video(model_path, output_dir="../videos", episode_length=500, fps=15):
    """Create video showing the CartPole swing-up and balancing"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path)
    
    # Create environment
    env = CartPoleSwingUpEnv()
    
    # Collect trajectory
    print(f"Collecting {episode_length} steps of trajectory...")
    obs, _ = env.reset()
    
    frames = []
    total_reward = 0
    swing_up_complete = False
    swing_up_step = None
    
    for step in range(episode_length):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        x, x_dot, theta, theta_dot = obs
        
        # Check if swing-up is complete
        angle_deg = np.rad2deg(theta)
        if not swing_up_complete and abs(angle_deg) < 15 and abs(theta_dot) < 1.0:
            swing_up_complete = True
            swing_up_step = step
            print(f"Swing-up completed at step {step}!")
        
        frames.append({
            'step': step,
            'x': x,
            'theta': theta,
            'action': action,
            'reward': reward,
            'total_reward': total_reward,
            'swing_up_complete': swing_up_complete
        })
        
        if step % 50 == 0:
            print(f"Step {step}: Angle = {angle_deg:.1f}°, Total reward = {total_reward:.1f}")
    
    env.close()
    
    # Prepare trajectory data for analysis
    trajectory = {
        'steps': [f['step'] for f in frames],
        'x': [f['x'] for f in frames],
        'theta': [f['theta'] for f in frames],
        'theta_dot': [obs[3] for obs in [env.reset()[0]] + [None]*(len(frames)-1)],  # Placeholder
        'actions': [f['action'] for f in frames],
        'rewards': [f['reward'] for f in frames]
    }
    
    # Create analysis plots
    create_analysis_plots(trajectory, output_dir)
    create_sequence_frames(trajectory, output_dir)
    
    # Create video
    print(f"Creating video from {len(frames)} frames...")
    
    # Video settings
    frame_width = 1200
    frame_height = 800
    frame_interval = 3  # Use every 3rd frame
    frames_to_use = frames[::frame_interval]
    
    output_path = os.path.join(output_dir, 'cartpole_demo.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    if not out.isOpened():
        print("Error: Could not create video writer")
        return None
    
    for i, data in enumerate(frames_to_use):
        # Create frame
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.patch.set_facecolor('white')
        
        x, theta = data['x'], data['theta']
        
        # Draw cart and pole
        cart_width, cart_height = 0.5, 0.25
        pole_length = 0.7
        
        # Cart
        cart_x = x - cart_width/2
        cart_y = -cart_height/2
        cart_rect = plt.Rectangle((cart_x, cart_y), cart_width, cart_height, 
                                 fill=True, color='#2E5BDA', alpha=0.9)
        ax.add_patch(cart_rect)
        
        # Pole
        pole_x = x + pole_length * np.sin(theta)
        pole_y = pole_length * np.cos(theta)
        ax.plot([x, pole_x], [cart_height/2, pole_y], 'r-', linewidth=12)
        ax.plot(pole_x, pole_y, 'ro', markersize=15)
        
        # Ground
        ax.axhline(y=-cart_height/2, color='black', linewidth=4)
        
        # Styling
        ax.set_xlim(-3.5, 3.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        
        # Title and info
        angle_deg = np.rad2deg(theta)
        phase = "BALANCING" if data['swing_up_complete'] else "SWINGING UP"
        
        ax.set_title(f'CartPole Swing-Up Demo | Step {data["step"]:03d}/{episode_length}\n' +
                    f'Phase: {phase} | Angle: {angle_deg:+6.1f}° | Total Reward: {data["total_reward"]:.1f}', 
                    fontsize=14, fontweight='bold')
        
        ax.grid(True, alpha=0.3)
        
        # Convert to video frame
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (frame_width, frame_height))
        
        out.write(img)
        plt.close()
        
        if i % 20 == 0:
            print(f"Video progress: {i+1}/{len(frames_to_use)} frames")
    
    out.release()
    cv2.destroyAllWindows()
    
    duration = len(frames_to_use) / fps
    print(f"✅ Video saved: {output_path}")
    print(f"📏 Duration: {duration:.1f} seconds")
    print(f"🎬 Frames: {len(frames_to_use)} at {fps} FPS")
    
    return output_path

def main():
    """Main function to create demo materials"""
    
    print("CartPole Demo Creator")
    print("=" * 25)
    
    # Find model
    models_to_try = [
        "../models/best_model.zip",
        "../models/cartpole_swingup_ppo.zip"
    ]
    
    model_path = None
    for model in models_to_try:
        if os.path.exists(model):
            model_path = model
            print(f"Found model: {model}")
            break
    
    if not model_path:
        print("No trained model found!")
        return
    
    try:
        # Create video and visualizations
        video_path = create_video(model_path, "../videos", episode_length=500, fps=15)
        
        if video_path:
            print(f"\n🎉 Demo materials created successfully!")
            print(f"📁 All outputs saved in videos/ directory")
        
    except Exception as e:
        print(f"Error creating demo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()