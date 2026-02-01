#!/usr/bin/env python3
"""
Trajectory Visualization Script for PlaNet RSSM

This script loads a trained RSSM model and visualizes:
1. Real trajectories from the environment
2. Imagined trajectories from the RSSM model
3. CEM planning trajectories
4. Comparison between real vs imagined dynamics
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from collections import defaultdict

# Import your models and environment
from models.RSSM import RSSM
from cem_planner import PlaNetController
from environment.dmc_env import DMCEnvironment

def load_model(checkpoint_path, device='cpu'):
    """Load RSSM model from checkpoint"""
    print(f"Loading model from {checkpoint_path}...")

    # Create RSSM model (same architecture as training)
    rssm = RSSM(
        state_size=30,
        action_size=6,
        observation_shape=(3, 64, 64),
        hidden_size=200,
        device=device
    ).to(device)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        rssm.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
        print(f"   Best eval return: {checkpoint.get('best_eval_return', 'unknown'):.2f}")
    else:
        # Handle old format (just state dict)
        rssm.load_state_dict(checkpoint)
        print("✅ Model loaded (legacy format)")

    rssm.eval()
    return rssm

def collect_real_trajectory(env, controller, max_steps=200, device='cpu'):
    """Collect a real trajectory using the controller"""
    print("🎬 Collecting real trajectory...")

    observations = []
    actions = []
    rewards = []
    states = []
    hidden_states = []

    obs, _ = env.reset()
    obs_tensor = torch.tensor(obs.copy(), dtype=torch.float32).permute(2, 0, 1) / 255.0

    controller.reset(obs_tensor)
    observations.append(obs_tensor.clone())

    for step in range(max_steps):
        # Get action from controller
        action = controller.act(obs_tensor)
        actions.append(torch.tensor(action, dtype=torch.float32))

        # Store current state and hidden state
        states.append(controller.current_state_belief.clone())
        hidden_states.append(controller.current_hidden.clone())

        # Take action in environment
        obs, reward, terminated, truncated, _ = env.step(action)
        obs_tensor = torch.tensor(obs.copy(), dtype=torch.float32).permute(2, 0, 1) / 255.0

        observations.append(obs_tensor.clone())
        rewards.append(reward)

        if terminated or truncated:
            print(f"   Episode ended at step {step}")
            break

    return {
        'observations': torch.stack(observations),
        'actions': torch.stack(actions),
        'rewards': torch.tensor(rewards),
        'states': torch.stack(states),
        'hidden_states': torch.stack(hidden_states)
    }

def generate_imagined_trajectory(rssm, initial_state, initial_hidden, actions, device='cpu'):
    """Generate imagined trajectory using RSSM model"""
    print("🧠 Generating imagined trajectory...")

    imagined_observations = []
    imagined_rewards = []
    current_state = initial_state
    current_hidden = initial_hidden

    with torch.no_grad():
        for action in actions:
            # Predict next state
            action_input = action.unsqueeze(0).to(device)

            # Encode state-action
            state_action_input = torch.cat([current_state, action_input], dim=-1)
            state_action_embedding = rssm.state_action(state_action_input)

            # Update hidden state
            rnn_input = torch.cat([state_action_embedding, current_hidden], dim=-1).unsqueeze(1)
            _, current_hidden = rssm.rnn(rnn_input)
            current_hidden = current_hidden.squeeze(1)

            # Sample next state (deterministic for visualization)
            current_state, _, _ = rssm.sample_prior(current_hidden, deterministic=True)

            # Generate observation and reward
            imagined_obs = rssm.decode(current_state)
            imagined_reward = rssm.reward(torch.cat([current_state, current_hidden], dim=-1))

            imagined_observations.append(imagined_obs.squeeze())
            imagined_rewards.append(imagined_reward.item())

    return {
        'observations': torch.stack(imagined_observations),
        'rewards': torch.tensor(imagined_rewards)
    }

def visualize_observations(real_obs, imagined_obs, timesteps, save_path=None):
    """Visualize real vs imagined observations"""
    print("📊 Creating observation comparison...")

    n_timesteps = min(len(timesteps), 8)  # Show up to 8 timesteps

    fig, axes = plt.subplots(2, n_timesteps, figsize=(n_timesteps*3, 6))
    fig.suptitle('Real vs Imagined Observations', fontsize=16)

    for i, t in enumerate(timesteps[:n_timesteps]):
        if t < len(real_obs) and t < len(imagined_obs):
            # Real observation
            real_img = real_obs[t].permute(1, 2, 0).numpy()
            axes[0, i].imshow(real_img)
            axes[0, i].set_title(f'Real t={t}')
            axes[0, i].axis('off')

            # Imagined observation
            imagined_img = imagined_obs[t].permute(1, 2, 0).numpy()
            imagined_img = np.clip(imagined_img, 0, 1)  # Ensure valid range
            axes[1, i].imshow(imagined_img)
            axes[1, i].set_title(f'Imagined t={t}')
            axes[1, i].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   Saved to {save_path}")
    plt.show()

def visualize_rewards_and_actions(real_trajectory, imagined_trajectory, save_path=None):
    """Visualize reward and action comparisons"""
    print("📈 Creating reward and action plots...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Trajectory Analysis', fontsize=16)

    # Reward comparison
    axes[0, 0].plot(real_trajectory['rewards'].numpy(), label='Real Rewards', alpha=0.8)
    axes[0, 0].plot(imagined_trajectory['rewards'].numpy(), label='Imagined Rewards', alpha=0.8)
    axes[0, 0].set_title('Rewards Over Time')
    axes[0, 0].set_xlabel('Timestep')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Cumulative rewards
    real_cumulative = np.cumsum(real_trajectory['rewards'].numpy())
    imagined_cumulative = np.cumsum(imagined_trajectory['rewards'].numpy())
    axes[0, 1].plot(real_cumulative, label='Real Cumulative', alpha=0.8)
    axes[0, 1].plot(imagined_cumulative, label='Imagined Cumulative', alpha=0.8)
    axes[0, 1].set_title('Cumulative Rewards')
    axes[0, 1].set_xlabel('Timestep')
    axes[0, 1].set_ylabel('Cumulative Reward')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Actions over time (show first 3 action dimensions)
    actions = real_trajectory['actions'].numpy()
    for i in range(min(3, actions.shape[1])):
        axes[1, 0].plot(actions[:, i], label=f'Action {i}', alpha=0.8)
    axes[1, 0].set_title('Actions Over Time')
    axes[1, 0].set_xlabel('Timestep')
    axes[1, 0].set_ylabel('Action Value')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # State norm over time
    state_norms = torch.norm(real_trajectory['states'], dim=1).numpy()
    axes[1, 1].plot(state_norms, label='State Magnitude', alpha=0.8)
    axes[1, 1].set_title('State Representation Magnitude')
    axes[1, 1].set_xlabel('Timestep')
    axes[1, 1].set_ylabel('L2 Norm')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   Saved to {save_path}")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Visualize PlaNet RSSM trajectories')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--steps', type=int, default=100,
                        help='Number of steps to simulate')
    parser.add_argument('--output-dir', type=str, default='visualizations',
                        help='Directory to save visualizations')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use (cpu/cuda)')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    rssm = load_model(args.checkpoint, device)

    # Create environment
    env = DMCEnvironment('walker', 'walk')
    action_dim = env.action_space.shape[0]

    # Create controller
    controller = PlaNetController(rssm, action_dim, horizon=4, action_repeat=2)

    # Collect real trajectory
    real_trajectory = collect_real_trajectory(env, controller, args.steps, device)

    # Generate imagined trajectory using same actions
    imagined_trajectory = generate_imagined_trajectory(
        rssm,
        real_trajectory['states'][0],
        real_trajectory['hidden_states'][0],
        real_trajectory['actions'],
        device
    )

    # Create visualizations
    print("🎨 Creating visualizations...")

    # Observation comparison at key timesteps
    timesteps = [0, 10, 20, 30, 40, 50, 60, 70]
    obs_save_path = os.path.join(args.output_dir, 'observations_comparison.png')
    visualize_observations(
        real_trajectory['observations'][1:],  # Skip initial obs
        imagined_trajectory['observations'],
        timesteps,
        obs_save_path
    )

    # Reward and action analysis
    metrics_save_path = os.path.join(args.output_dir, 'trajectory_metrics.png')
    visualize_rewards_and_actions(real_trajectory, imagined_trajectory, metrics_save_path)

    # Print summary statistics
    print("\n📊 TRAJECTORY SUMMARY:")
    print(f"   Real episode return: {real_trajectory['rewards'].sum():.2f}")
    print(f"   Imagined episode return: {imagined_trajectory['rewards'].sum():.2f}")
    print(f"   Trajectory length: {len(real_trajectory['rewards'])} steps")
    print(f"   Average real reward: {real_trajectory['rewards'].mean():.4f}")
    print(f"   Average imagined reward: {imagined_trajectory['rewards'].mean():.4f}")
    print(f"   Reward prediction error (MAE): {torch.abs(real_trajectory['rewards'] - imagined_trajectory['rewards']).mean():.4f}")

    env.close()
    print(f"\n✅ Visualization complete! Check {args.output_dir}/ for results.")

if __name__ == "__main__":
    main()