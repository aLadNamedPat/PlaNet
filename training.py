import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from dm_control import suite
from dm_control.suite.wrappers import pixels
import os

def setup_headless_rendering():
    """Setup headless rendering for DMC environments"""
    # Try different rendering backends
    backends = [
        ('egl', 'egl'),
        ('osmesa', 'osmesa'),
        ('glfw', 'glfw')
    ]

    for mujoco_gl, pyopengl_platform in backends:
        try:
            os.environ['MUJOCO_GL'] = mujoco_gl
            os.environ['PYOPENGL_PLATFORM'] = pyopengl_platform
            print(f"Trying rendering backend: {mujoco_gl}")
            return
        except Exception as e:
            print(f"Backend {mujoco_gl} failed: {e}")
            continue

    print("Warning: Could not set up any rendering backend")

setup_headless_rendering()
from models.RSSM import RSSM
from collections import deque
import random
from torch.distributions import Normal, kl_divergence
import wandb
from cem_planner import PlaNetController

def create_dmc_env_safe(domain_name="walker", task_name="walk", height=64, width=64, camera_id=0):
    """Safely create DMC environment with rendering fallbacks"""
    backends = ['egl', 'osmesa', 'glfw']

    for backend in backends:
        try:
            print(f"Attempting to create DMC environment with {backend} backend...")
            os.environ['MUJOCO_GL'] = backend

            # Create base environment
            env = suite.load(domain_name=domain_name, task_name=task_name)

            # Try to wrap with pixels
            env = pixels.Wrapper(
                env,
                pixels_only=True,
                render_kwargs={'height': height, 'width': width, 'camera_id': camera_id}
            )

            # Test if rendering works by calling reset
            time_step = env.reset()
            _ = time_step.observation['pixels']  # This will fail if rendering doesn't work

            print(f"Successfully created DMC environment with {backend} backend")
            return env

        except Exception as e:
            print(f"Backend {backend} failed: {e}")
            continue

    raise RuntimeError("Could not create DMC environment with any rendering backend. "
                       "Consider using a different environment or running on a system with graphics support.")

class DMCWrapper:
    """Wrapper to make DMC environment compatible with Gymnasium interface"""
    def __init__(self, dmc_env):
        self.env = dmc_env
        self.action_spec = dmc_env.action_spec()
        self.observation_spec = dmc_env.observation_spec()

    @property
    def action_space(self):
        """Mock action space for compatibility"""
        class ActionSpace:
            def __init__(self, action_spec):
                self.shape = (len(action_spec.minimum),)
                self.low = action_spec.minimum
                self.high = action_spec.maximum

            def sample(self):
                return np.random.uniform(self.low, self.high)

        return ActionSpace(self.action_spec)

    @property
    def observation_space(self):
        """Mock observation space for compatibility"""
        class ObservationSpace:
            def __init__(self, obs_spec):
                if 'pixels' in obs_spec:
                    self.shape = obs_spec['pixels'].shape
                else:
                    # Fallback for other observation types
                    self.shape = (64, 64, 3)

        return ObservationSpace(self.observation_spec)

    def reset(self):
        time_step = self.env.reset()
        obs = time_step.observation['pixels']
        return obs, {}

    def step(self, action):
        time_step = self.env.step(action)
        obs = time_step.observation['pixels']
        reward = time_step.reward or 0.0
        terminated = time_step.last()
        truncated = False  # DMC doesn't distinguish between terminated and truncated
        info = {}
        return obs, reward, terminated, truncated, info

    def close(self):
        self.env.close()

class ExperienceBuffer:
    def __init__(self, max_size=100000):
        """Store episodes for training"""
        self.observations = []
        self.actions = []
        self.rewards = []
        self.episode_lengths = []
        self.max_size = max_size

    def add_episode(self, obs_seq, action_seq, reward_seq):
        """Add a complete episode to the buffer"""
        self.observations.extend(obs_seq)
        self.actions.extend(action_seq)
        self.rewards.extend(reward_seq)
        self.episode_lengths.append(len(obs_seq))

        # Remove oldest episodes if buffer is too large
        while len(self.observations) > self.max_size:
            oldest_episode_len = self.episode_lengths.pop(0)
            self.observations = self.observations[oldest_episode_len:]
            self.actions = self.actions[oldest_episode_len:]
            self.rewards = self.rewards[oldest_episode_len:]

    def get_random_sequences(self, batch_size, sequence_length):
        """Sample random sequences of length L for training"""
        if len(self.observations) < sequence_length:
            return None, None, None

        obs_batch = []
        action_batch = []
        reward_batch = []

        for _ in range(batch_size):
            # Find valid start indices (ensuring we have enough steps)
            valid_start = len(self.observations) - sequence_length
            start_idx = random.randint(0, valid_start)

            obs_seq = self.observations[start_idx:start_idx + sequence_length]
            action_seq = self.actions[start_idx:start_idx + sequence_length]
            reward_seq = self.rewards[start_idx:start_idx + sequence_length]

            obs_batch.append(obs_seq)
            action_batch.append(action_seq)
            reward_batch.append(reward_seq)

        return torch.stack([torch.stack(seq) for seq in obs_batch]), \
               torch.stack([torch.stack(seq) for seq in action_batch]), \
               torch.stack([torch.stack(seq) for seq in reward_batch])

    def merge_buffer(self, other_buffer):
        """Merge another buffer into this one"""
        self.observations.extend(other_buffer.observations)
        self.actions.extend(other_buffer.actions)
        self.rewards.extend(other_buffer.rewards)
        self.episode_lengths.extend(other_buffer.episode_lengths)

        # Maintain size limit
        while len(self.observations) > self.max_size:
            oldest_episode_len = self.episode_lengths.pop(0)
            self.observations = self.observations[oldest_episode_len:]
            self.actions = self.actions[oldest_episode_len:]
            self.rewards = self.rewards[oldest_episode_len:]

def collect_random_episodes(env, num_episodes, max_steps_per_episode=1000):
    """Collect S random seed episodes for initial dataset"""
    buffer = ExperienceBuffer()

    print(f"Collecting {num_episodes} random episodes...")

    for episode in range(num_episodes):
        obs_sequence = []
        action_sequence = []
        reward_sequence = []

        obs, info = env.reset()
        # Convert image from (H, W, C) to (C, H, W) and normalize to [0, 1]
        obs_tensor = torch.tensor(obs, dtype=torch.float32).permute(2, 0, 1) / 255.0
        obs_sequence.append(obs_tensor)

        for step in range(max_steps_per_episode):
            # Random action sampling
            action = env.action_space.sample()
            action_tensor = torch.tensor(action, dtype=torch.float32)
            action_sequence.append(action_tensor)

            obs, reward, terminated, truncated, info = env.step(action)
            # Convert image from (H, W, C) to (C, H, W) and normalize to [0, 1]
            obs_tensor = torch.tensor(obs, dtype=torch.float32).permute(2, 0, 1) / 255.0
            obs_sequence.append(obs_tensor)
            reward_sequence.append(torch.tensor(reward, dtype=torch.float32))

            if terminated or truncated:
                break

        # Add episode to buffer (excluding last observation for action alignment)
        buffer.add_episode(obs_sequence[:-1], action_sequence, reward_sequence)

        if (episode + 1) % 10 == 0:
            print(f"Collected {episode + 1}/{num_episodes} episodes")

    print(f"Dataset collection complete. Total timesteps: {len(buffer.observations)}")
    return buffer

def compute_losses(rssm_output, reconstructed_obs, target_obs, predicted_rewards, target_rewards):
    """Compute RSSM training losses"""
    # Unpack RSSM outputs
    prior_states, posterior_states, hiddens, prior_mus, prior_logvars, posterior_mus, posterior_logvars, rewards = rssm_output

    # Reconstruction loss
    reconstruction_loss = nn.MSELoss()(reconstructed_obs, target_obs)

    # Reward prediction loss (MSE with assumed variance of 1)
    reward_loss = nn.MSELoss()(predicted_rewards.squeeze(-1), target_rewards)

    prior_dist = Normal(prior_mus, torch.exp(0.5 * prior_logvars))
    posterior_dist = Normal(posterior_mus, torch.exp(0.5 * posterior_logvars))
    kl_loss = kl_divergence(posterior_dist, prior_dist).sum(dim=-1).mean()

    return reconstruction_loss, reward_loss, kl_loss

def evaluate_controller(rssm, env, num_episodes=5, max_steps=1000):
    """
    Evaluate the trained RSSM with CEM planning

    Args:
        rssm: Trained RSSM model
        env: Environment to test in
        num_episodes: Number of episodes to run
        max_steps: Maximum steps per episode

    Returns:
        avg_return: Average return over episodes
    """
    action_dim = env.action_space.shape[0]
    controller = PlaNetController(rssm, action_dim, horizon=12, action_repeat=2)

    episode_returns = []

    for episode in range(num_episodes):
        obs, info = env.reset()
        obs_tensor = torch.tensor(obs, dtype=torch.float32)

        controller.reset(obs_tensor)
        episode_return = 0.0

        for step in range(max_steps):
            # Get action from CEM planner
            action = controller.act(obs_tensor)

            # Take action in environment
            obs, reward, terminated, truncated, info = env.step(action)
            obs_tensor = torch.tensor(obs, dtype=torch.float32)

            episode_return += reward

            # Update controller's internal state
            controller.update_state(action)

            if terminated or truncated:
                break

        episode_returns.append(episode_return)
        print(f"Episode {episode + 1}: Return = {episode_return:.2f}")

    avg_return = np.mean(episode_returns)
    std_return = np.std(episode_returns)

    print(f"\nEvaluation Results:")
    print(f"Average Return: {avg_return:.2f} ± {std_return:.2f}")

    return avg_return, episode_returns

def collect_cem_episodes(rssm, env, num_episodes=5, max_steps=1000, action_repeat=2):
    """
    Collect episodes using CEM planning for dataset augmentation

    Args:
        rssm: Trained RSSM model
        env: Environment
        num_episodes: Number of episodes to collect
        max_steps: Maximum steps per episode
        action_repeat: Action repeat parameter (R in paper)

    Returns:
        ExperienceBuffer with CEM-generated data
    """
    action_dim = env.action_space.shape[0]
    controller = PlaNetController(rssm, action_dim, horizon=12, action_repeat=action_repeat)

    buffer = ExperienceBuffer()

    print(f"Collecting {num_episodes} CEM-planned episodes...")

    for episode in range(num_episodes):
        obs_sequence = []
        action_sequence = []
        reward_sequence = []

        obs, info = env.reset()
        obs_tensor = torch.tensor(obs, dtype=torch.float32).permute(2, 0, 1) / 255.0
        obs_sequence.append(obs_tensor)

        controller.reset(obs_tensor)

        for step in range(max_steps):
            # Get action from CEM planner (every R timesteps)
            action = controller.act(obs_tensor)
            action_tensor = torch.tensor(action, dtype=torch.float32)
            action_sequence.append(action_tensor)

            # Take action in environment
            obs, reward, terminated, truncated, info = env.step(action)
            obs_tensor = torch.tensor(obs, dtype=torch.float32).permute(2, 0, 1) / 255.0
            obs_sequence.append(obs_tensor)
            reward_sequence.append(torch.tensor(reward, dtype=torch.float32))

            # Update controller's internal state
            controller.update_state(action)

            if terminated or truncated:
                break

        # Add episode to buffer (excluding last observation for action alignment)
        buffer.add_episode(obs_sequence[:-1], action_sequence, reward_sequence)

        if (episode + 1) % 5 == 0:
            print(f"Collected {episode + 1}/{num_episodes} CEM episodes")

    print(f"CEM data collection complete. Collected {len(buffer.observations)} timesteps")
    return buffer

def train_rssm(S=5, B=32, L=50, num_epochs=100, learning_rate=1e-3,
               evaluate_every=50, evaluation_episodes=3,
               plan_every=25, planning_episodes=3, action_repeat=2):
    """
    Main training loop for RSSM
    S: Number of random seed episodes
    B: Batch size
    L: Sequence length for training chunks
    """

    # Initialize DMC walker environment with pixel observations (as per PlaNet paper)
    try:
        dmc_env = create_dmc_env_safe(domain_name="walker", task_name="walk", height=64, width=64, camera_id=0)
        env = DMCWrapper(dmc_env)
    except RuntimeError as e:
        print(f"DMC environment creation failed: {e}")
        print("Falling back to BipedalWalker-v3...")
        # Fallback to original BipedalWalker
        import gymnasium as gym
        from gymnasium.wrappers import AddRenderObservation
        env = gym.make("BipedalWalker-v3", render_mode="rgb_array")
        env = AddRenderObservation(env, render_only=True)

    # Environment specifications
    obs_shape = env.observation_space.shape 
    action_dim = env.action_space.shape[0]

    print(f"Observation shape: {obs_shape}")
    print(f"Action dimension: {action_dim}")

    # RSSM hyperparameters
    encoded_size = 1024
    latent_size = 30
    hidden_size = 200
    sa_dim = 200
    lstm_layers = 1

    wandb.init(
        project="planet-rssm-dmc-walker",
        config={
            "S": S,
            "B": B,
            "L": L,
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "encoded_size": encoded_size,
            "latent_size": latent_size,
            "hidden_size": hidden_size,
            "sa_dim": sa_dim,
            "lstm_layers": lstm_layers,
            "obs_shape": obs_shape,
            "action_dim": action_dim,
            "plan_every": plan_every,
            "planning_episodes": planning_episodes,
            "action_repeat": action_repeat,
            "environment": "DMC-walker-walk"
        }
    )

    # Initialize RSSM
    rssm = RSSM(
        action_size=action_dim,
        sa_dim=sa_dim,
        latent_size=latent_size,
        encoded_size=encoded_size,
        hidden_size=hidden_size,
        lstm_layers=lstm_layers
    )

    optimizer = optim.Adam(rssm.parameters(), lr=learning_rate)

    # Collect initial dataset
    dataset = collect_random_episodes(env, S)

    # Log dataset statistics
    wandb.log({
        "dataset_size": len(dataset.observations),
        "num_episodes": S,
        "avg_episode_length": len(dataset.observations) / S if S > 0 else 0
    })

    print(f"\nStarting training with:")
    print(f"- Batch size: {B}")
    print(f"- Sequence length: {L}")
    print(f"- Number of epochs: {num_epochs}")

    # Training loop
    for epoch in range(num_epochs):
        # Sample batch of sequences
        obs_batch, action_batch, reward_batch = dataset.get_random_sequences(B, L)

        if obs_batch is None:
            print("Not enough data for training batch, skipping...")
            continue

        optimizer.zero_grad()

        # Encode observations
        batch_size, seq_len, obs_size = obs_batch.shape
        encoded_obs = rssm.encode(obs_batch.view(-1, obs_size))
        encoded_obs = encoded_obs.view(batch_size, seq_len, -1)

        # Initialize states
        prev_state = torch.zeros(batch_size, latent_size)
        prev_hidden = torch.zeros(batch_size, hidden_size)

        rssm_output = rssm.pass_through(prev_state, prev_hidden, encoded_obs, action_batch)
        prior_states, posterior_states, hiddens, _, _, _, _, predicted_rewards = rssm_output
        reconstructed = []

        for t in range(seq_len):
            decoded = rssm.decode(hiddens[:, t, :], posterior_states[:, t, :])
            reconstructed.append(decoded)

        reconstructed_obs = torch.stack(reconstructed, dim=1)

        reconstruction_loss, reward_loss, kl_loss = compute_losses(
            rssm_output, reconstructed_obs, obs_batch, predicted_rewards, reward_batch
        )

        total_loss = reconstruction_loss + reward_loss + kl_loss

        total_loss.backward()
        optimizer.step()

        wandb.log({
            "epoch": epoch,
            "total_loss": total_loss.item(),
            "reconstruction_loss": reconstruction_loss.item(),
            "reward_loss": reward_loss.item(),
            "kl_loss": kl_loss.item(),
            "learning_rate": learning_rate
        })

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Total Loss: {total_loss.item():.4f}, "
                  f"Reconstruction: {reconstruction_loss.item():.4f}, "
                  f"Reward: {reward_loss.item():.4f}, "
                  f"KL: {kl_loss.item():.4f}")

        if epoch % evaluate_every == 0 and epoch > 0:
            print(f"\n=== Evaluating Controller at Epoch {epoch} ===")
            rssm.eval()  # Set to evaluation mode

            avg_return, episode_returns = evaluate_controller(
                rssm, env, num_episodes=evaluation_episodes, max_steps=1000
            )

            # Log evaluation results to wandb
            wandb.log({
                "eval_avg_return": avg_return,
                "eval_std_return": np.std(episode_returns),
                "eval_epoch": epoch
            })

            rssm.train()  # Back to training mode
            print("=== Evaluation Complete ===\n")

        # CEM Planning and dataset augmentation
        if epoch % plan_every == 0 and epoch > 0:
            print(f"\n=== CEM Planning and Data Collection at Epoch {epoch} ===")
            rssm.eval()  # Set to evaluation mode for planning

            # Collect new data using CEM planning
            cem_buffer = collect_cem_episodes(
                rssm, env, num_episodes=planning_episodes,
                max_steps=1000, action_repeat=action_repeat
            )

            # Merge CEM data into training dataset
            dataset.merge_buffer(cem_buffer)

            # Log dataset statistics
            wandb.log({
                "dataset_size_after_planning": len(dataset.observations),
                "cem_episodes_added": planning_episodes,
                "planning_epoch": epoch
            })

            rssm.train()  # Back to training mode
            print(f"=== Planning Complete. Dataset now has {len(dataset.observations)} timesteps ===\n")

    # Save model and log as wandb artifact
    model_path = 'trained_rssm_dmc_walker.pth'
    torch.save(rssm.state_dict(), model_path)

    # Create wandb artifact for model
    artifact = wandb.Artifact("rssm_model", type="model")
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)

    env.close()
    wandb.finish()
    return rssm

if __name__ == "__main__":
    # Train the model with CEM evaluation and planning
    trained_rssm = train_rssm(
        S=10, B=50, L=50,
        num_epochs= 100,
        evaluate_every=50,
        evaluation_episodes=3,
        plan_every=25,  # CEM planning every 25 epochs
        planning_episodes=3,  # Collect 3 episodes per planning phase
        action_repeat=2  # Action repeat parameter R
    )

    print("Training complete! Model saved as 'trained_rssm_dmc_walker.pth'")

    # Final evaluation
    print("\n=== Final Evaluation ===")
    try:
        final_dmc_env = create_dmc_env_safe(domain_name="walker", task_name="walk", height=64, width=64, camera_id=0)
        final_eval_env = DMCWrapper(final_dmc_env)
    except RuntimeError as e:
        print(f"DMC environment creation failed for evaluation: {e}")
        print("Using BipedalWalker-v3 for final evaluation...")
        final_eval_env = gym.make("BipedalWalker-v3", render_mode="rgb_array")
        from gymnasium.wrappers import AddRenderObservation
        final_eval_env = AddRenderObservation(final_eval_env, render_only=True)

    final_avg_return, _ = evaluate_controller(trained_rssm, final_eval_env, num_episodes=10)
    final_eval_env.close()
    print(f"Final Average Return: {final_avg_return:.2f}")