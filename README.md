# CartPole Swing-Up with PPO

A reinforcement learning project that trains a PPO (Proximal Policy Optimization) agent to perform the CartPole swing-up task using Stable-Baselines3 and Gymnasium.

## Overview

This project implements a custom CartPole swing-up environment where the pole starts hanging down and the agent must learn to swing it up and balance it in the upright position. The environment uses realistic physics and provides shaped rewards to encourage the swing-up behavior.

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for fast Python package management. If you don't have uv installed, you can install it using:

```bash
# macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or with pip
pip install uv
```

### Project Setup

1. Clone the repository:
```bash
git clone https://github.com/nimninja/CartPole_25.git
cd CartPole_25
```

2. Install dependencies:
```bash
uv sync
```

3. Activate the virtual environment:
```bash
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate     # Windows
```

## Usage

### Training a New Model

To train a new PPO agent on the CartPole swing-up task:

```bash
python src/train_cartpole_swingup.py
```

The training script will:
- Create a vectorized environment with 16 parallel environments
- Train for 2 million timesteps
- Save evaluation logs and the best model
- Display progress with a progress bar

### Testing a Trained Model

To test a trained model (uncomment the test line in the script):

```bash
# Edit src/train_cartpole_swingup.py and uncomment:
# test_trained_model()
```

### Visualizing Training Progress

The project includes utility functions for analysis:

```python
# Plot training progress
plot_training_progress()

# Analyze solution strategy
analyze_solution_strategy()
```

## Project Structure

```
CartPole_25/
├── src/                          # Source code
│   ├── train_cartpole_swingup.py # Main training script
│   ├── fullvirenv.py             # old script
│   ├── realenv.py                # ??
│   ├── visualize.py              # Visualization (may not be very helpful)
│   └── ...
├── models/                       # Trained models and checkpoints
├── tensorboard/                  # TensorBoard logs
├── pyproject.toml                # Project dependencies
├── uv.lock                       # Dependency lock file
└── README.md                     # This file
```

## Environment Details

The CartPole swing-up environment features:

- **Physics**: Realistic cartpole dynamics with gravity, mass, and friction
- **Observation Space**: [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
- **Action Space**: Discrete(2) - move cart left or right
- **Reward Function**: 
  - Primary: cosine of pole angle (1 when upright, -1 when hanging)
  - Bonus: +2.0 when pole is within ±12° of upright
  - Penalties: Small penalties for excessive velocities and cart position

## Hyperparameters

The PPO agent uses the following optimized hyperparameters:
- Learning rate: 3e-4
- Steps per rollout: 2048
- Batch size: 64
- Training epochs per update: 10
- Discount factor (γ): 0.99
- GAE lambda: 0.95
- Clip range: 0.2
- Entropy coefficient: 0.01

## Requirements

- Python ≥ 3.10
- stable-baselines3[extra] ≥ 2.6.0
- matplotlib ≥ 3.10.3
