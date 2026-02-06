import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sys
import os
from collections import deque
import time

# Set up rendering BEFORE any dm_control imports
os.environ['MUJOCO_GL'] = 'osmesa'
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'

sys.path.insert(0, '/home/grokveritas/PlaNet')

from dm_control import suite
from dm_control.suite.wrappers import pixels
from cem_planner import PlaNetController


def create_walker_env():
    """Create Walker environment with pixel observations"""
    env = suite.load(domain_name="walker", task_name="walk")
    env = pixels.Wrapper(
        env,
        pixels_only=True,
        render_kwargs={'height': 64, 'width': 64, 'camera_id': 0}
    )
    return env


def load_rssm(checkpoint_path, device='cpu'):
    """
    Load RSSM from checkpoint.
    
    Args:
        checkpoint_path: Path to RSSM checkpoint
        device: torch device
    """
    from models.RSSM import RSSM
    
    # RSSM hyperparameters (must match training)
    action_dim = 6
    encoded_size = 1024
    latent_size = 30
    hidden_size = 200
    
    rssm = RSSM(
        action_size=action_dim,
        latent_size=latent_size,
        encoded_size=encoded_size,
        hidden_size=hidden_size,
        min_std_dev=0.1,
        device=device
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    rssm.load_state_dict(checkpoint['model_state_dict'])
    rssm.eval()
    
    # Print checkpoint info
    if 'epoch' in checkpoint:
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    if 'best_eval_return' in checkpoint:
        print(f"Best eval return at save: {checkpoint['best_eval_return']:.2f}")
    
    return rssm


# PlaNetController is now imported from cem_planner.py


def run_episode_and_collect_actions(env, rssm, device='cpu', max_steps=500,
                                     horizon=12):
    """
    Run one episode using CEM planning and collect all actions.
    
    Returns:
        actions: numpy array [T, action_dim]
        rewards: numpy array [T]
        frames: list of numpy arrays for video
        plan_stats_history: list of planning statistics per step
    """
    action_dim = 6
    
    controller = PlaNetController(
        rssm, action_dim,
        horizon=horizon,
        action_repeat=2  # Use correct interface from cem_planner.py
    )
    
    actions_list = []
    rewards_list = []
    frames_list = []
    plan_stats_history = []
    
    # Reset environment
    time_step = env.reset()
    obs = time_step.observation['pixels']
    obs_tensor = torch.tensor(obs.copy(), dtype=torch.float32).permute(2, 0, 1) / 255.0
    
    # Save first frame
    frames_list.append(obs.copy())
    
    # Initialize controller
    controller.reset(obs_tensor)
    
    print(f"Running episode with CEM planning (horizon={horizon})...")
    total_reward = 0
    for step in range(max_steps):
        if step % 50 == 0:
            print(f"  Step {step}/{max_steps}...")
        
        # Get action from CEM planner (now uses current observation)
        action = controller.act(obs_tensor)

        # Clip action to valid range
        action = np.clip(action, -1.0, 1.0)
        actions_list.append(action.copy())
        plan_stats_history.append({})  # No planning stats from correct controller

        # Step environment
        time_step = env.step(action)
        reward = time_step.reward if time_step.reward is not None else 0.0
        rewards_list.append(reward)
        total_reward += reward
        
        # Get next observation
        obs = time_step.observation['pixels']
        obs_tensor = torch.tensor(obs.copy(), dtype=torch.float32).permute(2, 0, 1) / 255.0
        frames_list.append(obs.copy())

        # Update controller state (now only uses action, like cem_planner)
        controller.update_state(action)
        
        if time_step.last():
            print(f"  Episode ended at step {step+1}")
            break
    
    return np.array(actions_list), np.array(rewards_list), frames_list, plan_stats_history


def visualize_actions(actions, rewards, frames, output_path='action_visualization.png',
                      action_names=None):
    """
    Create comprehensive action visualization.
    
    Args:
        actions: [T, action_dim] numpy array
        rewards: [T] numpy array
        frames: list of frames
        output_path: where to save the figure
        action_names: optional list of names for each action dimension
    """
    T, action_dim = actions.shape
    
    if action_names is None:
        # Walker action names (from dm_control walker)
        action_names = [
            'Right hip',
            'Right knee', 
            'Right ankle',
            'Left hip',
            'Left knee',
            'Left ankle'
        ]
        if len(action_names) != action_dim:
            action_names = [f'Action {i}' for i in range(action_dim)]
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(4, 3, figure=fig, height_ratios=[1, 1, 1, 0.8])
    
    timesteps = np.arange(T)
    colors = plt.cm.tab10(np.linspace(0, 1, action_dim))
    
    # ============================================================
    # Plot 1: All actions over time (top row, spans 2 columns)
    # ============================================================
    ax1 = fig.add_subplot(gs[0, :2])
    for i in range(action_dim):
        ax1.plot(timesteps, actions[:, i], label=action_names[i], 
                 color=colors[i], alpha=0.8, linewidth=1.5)
    ax1.set_xlabel('Timestep')
    ax1.set_ylabel('Action Value')
    ax1.set_title('All Actions Over Time (CEM Planning)')
    ax1.legend(loc='upper right', ncol=2, fontsize=8)
    ax1.set_ylim(-1.1, 1.1)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.grid(True, alpha=0.3)
    
    # ============================================================
    # Plot 2: Reward over time (top right)
    # ============================================================
    ax2 = fig.add_subplot(gs[0, 2])
    cumulative_reward = np.cumsum(rewards)
    ax2.plot(timesteps, rewards, color='green', linewidth=1.5, label='Instant')
    ax2.plot(timesteps, cumulative_reward, color='darkgreen', linewidth=1.5, 
             linestyle='--', alpha=0.7, label='Cumulative')
    ax2.set_xlabel('Timestep')
    ax2.set_ylabel('Reward')
    ax2.set_title(f'Reward Over Time\nTotal: {rewards.sum():.2f}')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # ============================================================
    # Plot 3-8: Individual action traces (middle rows)
    # ============================================================
    for i in range(min(action_dim, 6)):
        row = 1 + i // 3
        col = i % 3
        ax = fig.add_subplot(gs[row, col])
        
        ax.plot(timesteps, actions[:, i], color=colors[i], linewidth=1.5)
        ax.fill_between(timesteps, actions[:, i], alpha=0.2, color=colors[i])
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Value')
        ax.set_title(f'{action_names[i]}')
        ax.set_ylim(-1.1, 1.1)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        mean_val = actions[:, i].mean()
        std_val = actions[:, i].std()
        ax.text(0.02, 0.98, f'μ={mean_val:.2f}, σ={std_val:.2f}', 
                transform=ax.transAxes, fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # ============================================================
    # Plot 9: Action distribution histogram (bottom row)
    # ============================================================
    ax9 = fig.add_subplot(gs[3, 0])
    for i in range(action_dim):
        ax9.hist(actions[:, i], bins=30, alpha=0.5, label=action_names[i], 
                 color=colors[i], density=True)
    ax9.set_xlabel('Action Value')
    ax9.set_ylabel('Density')
    ax9.set_title('Action Value Distributions')
    ax9.legend(fontsize=6, ncol=2)
    ax9.grid(True, alpha=0.3)
    
    # ============================================================
    # Plot 10: Action correlation matrix
    # ============================================================
    ax10 = fig.add_subplot(gs[3, 1])
    corr_matrix = np.corrcoef(actions.T)
    im = ax10.imshow(corr_matrix, cmap='RdBu', vmin=-1, vmax=1)
    ax10.set_xticks(range(action_dim))
    ax10.set_yticks(range(action_dim))
    ax10.set_xticklabels([f'A{i}' for i in range(action_dim)], fontsize=8)
    ax10.set_yticklabels([f'A{i}' for i in range(action_dim)], fontsize=8)
    ax10.set_title('Action Correlation')
    plt.colorbar(im, ax=ax10, shrink=0.8)
    
    # Add correlation values
    for i in range(action_dim):
        for j in range(action_dim):
            text = ax10.text(j, i, f'{corr_matrix[i, j]:.1f}',
                           ha='center', va='center', fontsize=6)
    
    # ============================================================
    # Plot 11: Action magnitude heatmap
    # ============================================================
    ax11 = fig.add_subplot(gs[3, 2])
    
    # Show action magnitudes as a heatmap over time
    action_magnitudes = np.abs(actions).T  # [action_dim, T]
    im2 = ax11.imshow(action_magnitudes, aspect='auto', cmap='viridis',
                      extent=[0, T, action_dim-0.5, -0.5])
    ax11.set_xlabel('Timestep')
    ax11.set_ylabel('Action Dimension')
    ax11.set_yticks(range(action_dim))
    ax11.set_yticklabels([f'A{i}' for i in range(action_dim)], fontsize=8)
    ax11.set_title('Action Magnitude Heatmap')
    plt.colorbar(im2, ax=ax11, shrink=0.8, label='|action|')
    
    plt.suptitle('PlaNet CEM Policy - Action Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved action visualization to: {output_path}")


def visualize_cem_planning(plan_stats_history, output_path='cem_planning_visualization.png'):
    """
    Visualize the CEM planning process over time.
    
    Shows how the action distribution evolves during planning iterations
    and how planned actions compare across timesteps.
    """
    if not plan_stats_history:
        print("No planning statistics to visualize")
        return
    
    T = len(plan_stats_history)
    action_dim = plan_stats_history[0]['best_action'].shape[0]
    
    # Extract data
    all_best_actions = np.array([ps['best_action'] for ps in plan_stats_history])
    all_final_stds = np.array([ps['final_std'][0] for ps in plan_stats_history])  # First action's std
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    colors = plt.cm.tab10(np.linspace(0, 1, action_dim))
    action_names = ['R.Hip', 'R.Knee', 'R.Ankle', 'L.Hip', 'L.Knee', 'L.Ankle']
    
    # ============================================================
    # Plot 1: Selected actions over episode
    # ============================================================
    ax = axes[0, 0]
    for i in range(action_dim):
        ax.plot(all_best_actions[:, i], label=action_names[i], color=colors[i], alpha=0.8)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Action Value')
    ax.set_title('CEM-Selected Actions Over Episode')
    ax.legend(fontsize=8, ncol=2)
    ax.set_ylim(-1.1, 1.1)
    ax.grid(True, alpha=0.3)
    
    # ============================================================
    # Plot 2: Planning uncertainty (std) over episode
    # ============================================================
    ax = axes[0, 1]
    for i in range(action_dim):
        ax.plot(all_final_stds[:, i], label=action_names[i], color=colors[i], alpha=0.8)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Action Std (uncertainty)')
    ax.set_title('CEM Planning Uncertainty Over Episode')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    
    # ============================================================
    # Plot 3: Sample a few timesteps - show CEM iteration convergence
    # ============================================================
    ax = axes[0, 2]
    sample_timesteps = [0, T//4, T//2, 3*T//4, min(T-1, T-1)]
    for t_idx, t in enumerate(sample_timesteps):
        if t < len(plan_stats_history):
            iteration_means = plan_stats_history[t]['iteration_means']  # [num_iter, horizon, action_dim]
            # Plot first action's mean across iterations for action 0
            ax.plot(iteration_means[:, 0, 0], label=f't={t}', alpha=0.8)
    ax.set_xlabel('CEM Iteration')
    ax.set_ylabel('Action 0 Mean')
    ax.set_title('CEM Convergence (Action 0, First in Horizon)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # ============================================================
    # Plot 4: Planned horizon at a single timestep
    # ============================================================
    ax = axes[1, 0]
    sample_t = T // 2
    if sample_t < len(plan_stats_history):
        final_mean = plan_stats_history[sample_t]['final_mean']  # [horizon, action_dim]
        final_std = plan_stats_history[sample_t]['final_std']
        horizon = final_mean.shape[0]
        
        for i in range(action_dim):
            ax.plot(range(horizon), final_mean[:, i], label=action_names[i], color=colors[i])
            ax.fill_between(range(horizon), 
                           final_mean[:, i] - final_std[:, i],
                           final_mean[:, i] + final_std[:, i],
                           color=colors[i], alpha=0.2)
    ax.set_xlabel('Planning Horizon Step')
    ax.set_ylabel('Planned Action Value')
    ax.set_title(f'Planned Action Sequence at t={sample_t}')
    ax.legend(fontsize=8, ncol=2)
    ax.set_ylim(-1.5, 1.5)
    ax.grid(True, alpha=0.3)
    
    # ============================================================
    # Plot 5: Action change rate (derivative)
    # ============================================================
    ax = axes[1, 1]
    action_changes = np.diff(all_best_actions, axis=0)
    for i in range(action_dim):
        ax.plot(np.abs(action_changes[:, i]), label=action_names[i], color=colors[i], alpha=0.7)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('|Δ Action|')
    ax.set_title('Action Change Rate')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    
    # ============================================================
    # Plot 6: Action value distribution across episode
    # ============================================================
    ax = axes[1, 2]
    bp = ax.boxplot([all_best_actions[:, i] for i in range(action_dim)],
                    labels=action_names, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel('Action Value')
    ax.set_title('Action Distribution (Box Plot)')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('CEM Planning Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved CEM planning visualization to: {output_path}")


def visualize_actions_with_frames(actions, rewards, frames, output_path='actions_with_frames.png',
                                   num_frame_samples=8):
    """
    Create visualization with sample frames alongside action traces.
    """
    T, action_dim = actions.shape
    
    # Sample frame indices evenly
    frame_indices = np.linspace(0, min(len(frames)-1, T-1), num_frame_samples).astype(int)
    
    fig, axes = plt.subplots(3, num_frame_samples, figsize=(20, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, action_dim))
    timesteps = np.arange(T)
    
    # Row 1: Sampled frames
    for i, idx in enumerate(frame_indices):
        axes[0, i].imshow(frames[idx])
        axes[0, i].set_title(f't={idx}', fontsize=10)
        axes[0, i].axis('off')
    
    # Row 2: Actions up to each frame
    for i, idx in enumerate(frame_indices):
        ax = axes[1, i]
        for j in range(action_dim):
            ax.plot(timesteps[:idx+1], actions[:idx+1, j], 
                   color=colors[j], alpha=0.7, linewidth=1)
        ax.set_ylim(-1.1, 1.1)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
        if i == 0:
            ax.set_ylabel('Action')
        ax.set_xlabel('t')
        ax.grid(True, alpha=0.3)
    
    # Row 3: Current action values (bar chart)
    action_names = ['R.Hip', 'R.Knee', 'R.Ankle', 'L.Hip', 'L.Knee', 'L.Ankle']
    for i, idx in enumerate(frame_indices):
        ax = axes[2, i]
        if idx < T:
            current_actions = actions[idx]
            bars = ax.bar(range(action_dim), current_actions, color=colors)
            ax.set_ylim(-1.1, 1.1)
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax.set_xticks(range(action_dim))
            ax.set_xticklabels([f'A{j}' for j in range(action_dim)], fontsize=8)
            if i == 0:
                ax.set_ylabel('Action Value')
    
    plt.suptitle('CEM Policy Actions with Environment Frames', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved frames+actions visualization to: {output_path}")


def generate_dream_video(rssm, device='cpu', dream_length=200, action_strategy='random',
                        checkpoint_path=None, output_path='dream_episode.mp4', fps=30,
                        seed_obs=None, use_cem=False, cem_horizon=12):
    """
    Generate a "dream" MP4 video where RSSM imagines/hallucinates a trajectory
    purely from its learned world model without real environment interaction.

    Args:
        rssm: Trained RSSM model
        device: torch device
        dream_length: Number of timesteps to dream for
        action_strategy: 'random', 'cem', or 'zero' for action generation
        checkpoint_path: Path to load model from (if None, assumes already loaded)
        output_path: Path to save the dream video
        fps: Video frames per second
        seed_obs: Optional seed observation tensor [3, 64, 64]. If None, creates random
        use_cem: Whether to use CEM planning for actions (overrides action_strategy)
        cem_horizon: Planning horizon if using CEM

    Returns:
        imagined_frames: List of imagined observation frames
        imagined_rewards: Array of imagined rewards
        actions_taken: Array of actions used in the dream
    """
    import imageio

    print(f"\n{'='*60}")
    print("GENERATING RSSM DREAM VIDEO")
    print(f"{'='*60}")
    print(f"Dream length: {dream_length} steps")
    print(f"Action strategy: {action_strategy}")
    print(f"Device: {device}")
    print(f"Output: {output_path}")

    # Load model if checkpoint provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading model from: {checkpoint_path}")
        rssm = load_rssm(checkpoint_path, device)

    rssm.eval()

    # Initialize state
    batch_size = 1
    latent_size = rssm.prior_mu.out_features  # 30
    hidden_size = rssm.rnn.hidden_size        # 200
    action_dim = 6

    print(f"Model dimensions: latent={latent_size}, hidden={hidden_size}, action={action_dim}")

    with torch.no_grad():
        # Initialize hidden state
        current_hidden = torch.zeros(batch_size, hidden_size, device=device)

        # Initialize stochastic state
        if seed_obs is not None:
            print("Using provided seed observation")
            # Use encoder to get initial state from observation
            seed_obs = seed_obs.to(device)
            if seed_obs.dim() == 3:  # [3, 64, 64] -> [1, 3, 64, 64]
                seed_obs = seed_obs.unsqueeze(0)
            encoded_obs = rssm.encode(seed_obs)
            current_state, _, _ = rssm.sample_posterior(current_hidden, encoded_obs, deterministic=True)
        else:
            print("Using random initial state")
            # Start from a random state drawn from prior
            current_state, _, _ = rssm.sample_prior(current_hidden, deterministic=False)

        # Initialize CEM planner if requested
        planner = None
        if use_cem or action_strategy == 'cem':
            from cem_planner import CEMPlanner
            planner = CEMPlanner(rssm, action_dim, horizon=cem_horizon, device=device)
            print(f"Using CEM planner with horizon {cem_horizon}")

        # Storage for dream sequence
        imagined_frames = []
        imagined_rewards = []
        actions_taken = []

        print(f"Starting dream sequence...")

        for t in range(dream_length):
            if t % 50 == 0:
                print(f"  Dream step {t}/{dream_length}...")

            # 1. Decode current state to get imagined observation
            combined_state = torch.cat([current_hidden, current_state], dim=-1)
            imagined_obs = rssm.decode(current_hidden, current_state)  # [1, 3, 64, 64]

            # Convert to numpy for storage (make sure values are in [0, 1] range)
            imagined_frame = torch.sigmoid(imagined_obs).squeeze(0).permute(1, 2, 0).cpu().numpy()
            imagined_frame = np.clip(imagined_frame * 255, 0, 255).astype(np.uint8)
            imagined_frames.append(imagined_frame)

            # 2. Predict reward for current state
            reward_input = torch.cat([current_state, current_hidden], dim=-1)
            imagined_reward = rssm.reward(reward_input).squeeze().cpu().item()
            imagined_rewards.append(imagined_reward)

            # 3. Choose action for next step
            if use_cem or action_strategy == 'cem':
                action = planner.plan(current_state, current_hidden)
            elif action_strategy == 'random':
                action = np.random.uniform(-1.0, 1.0, size=action_dim)
            elif action_strategy == 'zero':
                action = np.zeros(action_dim)
            elif action_strategy == 'sine':
                # Sinusoidal actions for more interesting movement
                action = 0.5 * np.sin(0.1 * t + np.arange(action_dim))
            else:
                raise ValueError(f"Unknown action strategy: {action_strategy}")

            actions_taken.append(action.copy())

            # 4. Transition to next state using RSSM dynamics
            action_tensor = torch.tensor(action, dtype=torch.float32, device=device).unsqueeze(0)

            # Embed state-action
            state_action_input = torch.cat([current_state, action_tensor], dim=-1)
            embedded = F.relu(rssm.fc_embed_state_action(state_action_input))

            # Update hidden state
            current_hidden = rssm.rnn(embedded, current_hidden)

            # Sample next stochastic state from prior (imagination step)
            current_state, _, _ = rssm.sample_prior(current_hidden, deterministic=False)

        # Convert to numpy arrays
        imagined_rewards = np.array(imagined_rewards)
        actions_taken = np.array(actions_taken)

        print(f"Dream complete! Generated {len(imagined_frames)} frames")
        print(f"Total imagined reward: {imagined_rewards.sum():.2f}")
        print(f"Mean reward per step: {imagined_rewards.mean():.4f}")

        # Create video frames
        print(f"Creating dream video...")
        video_frames = []

        for t in range(len(imagined_frames)):
            # Create figure with imagined observation and action bars
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))

            # Left: Imagined observation
            axes[0].imshow(imagined_frames[t])
            axes[0].set_title(f'RSSM Dream - Step {t}', fontsize=12)
            axes[0].axis('off')

            # Right: Current action
            colors = plt.cm.tab10(np.linspace(0, 1, action_dim))
            action_names = ['R.Hip', 'R.Knee', 'R.Ankle', 'L.Hip', 'L.Knee', 'L.Ankle']

            current_action = actions_taken[t]
            bars = axes[1].bar(range(action_dim), current_action, color=colors)
            axes[1].set_ylim(-1.1, 1.1)
            axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            axes[1].set_xticks(range(action_dim))
            axes[1].set_xticklabels(action_names, fontsize=9)
            axes[1].set_ylabel('Action Value', fontsize=10)
            axes[1].grid(True, alpha=0.3, axis='y')

            # Add reward information
            cum_reward = imagined_rewards[:t+1].sum()
            axes[1].set_title(f'Action | Reward: {cum_reward:.1f}', fontsize=12)

            plt.tight_layout()

            # Convert to image array
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
            video_frames.append(img)
            plt.close(fig)

            if (t + 1) % 50 == 0:
                print(f"  Processed {t+1}/{len(imagined_frames)} frames...")

        # Save video
        print(f"Saving dream video to {output_path}...")
        imageio.mimsave(output_path, video_frames, fps=fps)

        print(f"Dream video saved! Duration: {len(video_frames)/fps:.1f} seconds")
        print(f"Action strategy used: {action_strategy}")

        return imagined_frames, imagined_rewards, actions_taken


def create_video(actions, rewards, frames, output_path='walker_policy.mp4', fps=30,
                  show_actions=True):
    """
    Create video directly without saving intermediate frames.
    
    Args:
        actions: [T, action_dim] numpy array
        rewards: [T] numpy array  
        frames: list of environment frames
        output_path: path for output MP4 file
        fps: frames per second
        show_actions: if True, show action bars alongside walker
    """
    import imageio
    
    T, action_dim = actions.shape
    colors = plt.cm.tab10(np.linspace(0, 1, action_dim))
    
    action_names = ['R.Hip', 'R.Knee', 'R.Ankle', 'L.Hip', 'L.Knee', 'L.Ankle']
    if len(action_names) != action_dim:
        action_names = [f'A{i}' for i in range(action_dim)]
    
    print(f"Generating video with {min(T, len(frames)-1)} frames...")
    
    video_frames = []
    
    for t in range(min(T, len(frames)-1)):
        if show_actions:
            # Create figure with walker and action bars
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            
            # Left: Environment frame
            axes[0].imshow(frames[t])
            axes[0].set_title(f't={t}', fontsize=12)
            axes[0].axis('off')
            
            # Right: Action bar chart
            current_actions = actions[t]
            bars = axes[1].bar(range(action_dim), current_actions, color=colors)
            axes[1].set_ylim(-1.1, 1.1)
            axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            axes[1].set_xticks(range(action_dim))
            axes[1].set_xticklabels(action_names, fontsize=9)
            axes[1].set_ylabel('Action', fontsize=10)
            axes[1].grid(True, alpha=0.3, axis='y')
            
            # Add cumulative reward
            cum_reward = rewards[:t+1].sum()
            axes[1].set_title(f'Reward: {cum_reward:.1f}', fontsize=12)
            
            plt.tight_layout()
            
            # Convert figure to image array
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]  # Remove alpha channel
            video_frames.append(img)
            plt.close(fig)
        else:
            # Just the walker frame (upscaled for better visibility)
            frame = frames[t]
            # Upscale from 64x64 to 256x256
            frame_large = np.repeat(np.repeat(frame, 4, axis=0), 4, axis=1)
            video_frames.append(frame_large)
        
        if (t + 1) % 100 == 0:
            print(f"  Processed {t+1}/{min(T, len(frames)-1)} frames...")
    
    # Write video
    print(f"Writing video to {output_path}...")
    imageio.mimsave(output_path, video_frames, fps=fps)
    print(f"Done! Video saved to: {output_path}")
    print(f"  Duration: {len(video_frames)/fps:.1f} seconds")
    print(f"  Total reward: {rewards.sum():.2f}")


def print_action_statistics(actions, rewards):
    """Print detailed statistics about the actions."""
    T, action_dim = actions.shape
    
    action_names = ['R.Hip', 'R.Knee', 'R.Ankle', 'L.Hip', 'L.Knee', 'L.Ankle']
    if len(action_names) != action_dim:
        action_names = [f'Action {i}' for i in range(action_dim)]
    
    print("\n" + "=" * 60)
    print("ACTION STATISTICS (CEM Planning)")
    print("=" * 60)
    print(f"Episode length: {T} timesteps")
    print(f"Total reward: {rewards.sum():.2f}")
    print(f"Mean reward per step: {rewards.mean():.4f}")
    print()
    
    print(f"{'Action':<12} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'|Mean|':>8}")
    print("-" * 60)
    for i in range(action_dim):
        a = actions[:, i]
        print(f"{action_names[i]:<12} {a.mean():>8.3f} {a.std():>8.3f} "
              f"{a.min():>8.3f} {a.max():>8.3f} {np.abs(a).mean():>8.3f}")
    
    print()
    print("Action correlations (Pearson):")
    corr = np.corrcoef(actions.T)
    
    # Print upper triangle
    print(f"{'':>12}", end='')
    for i in range(action_dim):
        print(f"{action_names[i]:>10}", end='')
    print()
    
    for i in range(action_dim):
        print(f"{action_names[i]:<12}", end='')
        for j in range(action_dim):
            if j >= i:
                print(f"{corr[i,j]:>10.2f}", end='')
            else:
                print(f"{'':>10}", end='')
        print()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize PlaNet CEM policy actions')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth',
                        help='Path to RSSM checkpoint file')
    parser.add_argument('--device', type=str, 
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--max-steps', type=int, default=500,
                        help='Maximum steps per episode')
    parser.add_argument('--num-episodes', type=int, default=1,
                        help='Number of episodes to run')
    parser.add_argument('--horizon', type=int, default=12,
                        help='CEM planning horizon')
    parser.add_argument('--output-dir', type=str, default='.',
                        help='Output directory for visualizations')
    parser.add_argument('--video-only', action='store_true',
                        help='Only generate video, skip static plots')
    parser.add_argument('--walker-only', action='store_true',
                        help='Video shows only walker (no action bars)')
    parser.add_argument('--fps', type=int, default=30,
                        help='Video frames per second')
    parser.add_argument('--dream-mode', action='store_true',
                        help='Generate dream video instead of real environment')
    parser.add_argument('--dream-length', type=int, default=200,
                        help='Number of timesteps for dream sequence')
    parser.add_argument('--action-strategy', type=str, default='random',
                        choices=['random', 'cem', 'zero', 'sine'],
                        help='Action strategy for dream generation')
    parser.add_argument('--cem-horizon', type=int, default=12,
                        help='CEM planning horizon for dream generation')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("PLANET CEM POLICY - ACTION VISUALIZATION")
    print(f"{'='*60}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {args.device}")
    print(f"Max steps: {args.max_steps}")
    print(f"CEM settings: horizon={args.horizon} (using cem_planner.py implementation)")

    # Load RSSM
    print("Loading RSSM model...")
    rssm = load_rssm(args.checkpoint, args.device)

    # Handle dream mode
    if args.dream_mode:
        print(f"\n--- DREAM MODE ---")
        print(f"Generating dream video with {args.dream_length} steps")
        print(f"Action strategy: {args.action_strategy}")

        # Generate dream video
        output_path = os.path.join(args.output_dir, f'dream_{args.action_strategy}_{args.dream_length}_steps.mp4')

        imagined_frames, imagined_rewards, actions_taken = generate_dream_video(
            rssm=rssm,
            device=args.device,
            dream_length=args.dream_length,
            action_strategy=args.action_strategy,
            output_path=output_path,
            fps=args.fps,
            use_cem=(args.action_strategy == 'cem'),
            cem_horizon=args.cem_horizon
        )

        # Print dream statistics
        print(f"\n--- DREAM STATISTICS ---")
        print(f"Dream length: {len(imagined_frames)} frames")
        print(f"Total imagined reward: {imagined_rewards.sum():.2f}")
        print(f"Mean reward per step: {imagined_rewards.mean():.4f}")
        print(f"Action statistics:")
        for i, action_name in enumerate(['R.Hip', 'R.Knee', 'R.Ankle', 'L.Hip', 'L.Knee', 'L.Ankle']):
            if i < actions_taken.shape[1]:
                action_vals = actions_taken[:, i]
                print(f"  {action_name}: mean={action_vals.mean():.3f}, std={action_vals.std():.3f}")

        print(f"\nDream video saved to: {output_path}")
        return  # Exit early for dream mode

    # Create environment (only for non-dream mode)
    print("\nCreating Walker environment...")
    env = create_walker_env()

    # Run episodes
    for ep in range(args.num_episodes):
        print(f"\n--- Episode {ep+1}/{args.num_episodes} ---")
        
        actions, rewards, frames, plan_stats_history = run_episode_and_collect_actions(
            env, rssm, args.device,
            max_steps=args.max_steps,
            horizon=args.horizon
        )
        
        print(f"Collected {len(actions)} actions, {len(frames)} frames")
        
        # Print statistics
        print_action_statistics(actions, rewards)
        
        # Create output paths
        output_base = os.path.join(args.output_dir, f'episode_{ep+1}')
        
        # Generate video
        create_video(
            actions, rewards, frames,
            output_path=f'{output_base}_walker.mp4',
            fps=args.fps,
            show_actions=not args.walker_only
        )
        
        # Generate static plots (unless video-only mode)
        if not args.video_only:
            visualize_actions(
                actions, rewards, frames,
                output_path=f'{output_base}_actions.png'
            )
            
            visualize_actions_with_frames(
                actions, rewards, frames,
                output_path=f'{output_base}_actions_frames.png'
            )
            
            visualize_cem_planning(
                plan_stats_history,
                output_path=f'{output_base}_cem_planning.png'
            )
    
    env.close()
    print("\nDone!")


if __name__ == "__main__":
    main()