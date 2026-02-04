import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import os
from torch.optim.lr_scheduler import CosineAnnealingLR

# Initialize environment setup for DMC BEFORE any dm_control imports
def initialize_dmc_environment():
    """Initialize DMC environment before any imports"""
    print("Pre-configuring environment for DMC...")

    # Check what's available
    import ctypes.util
    import subprocess

    osmesa_available = ctypes.util.find_library('OSMesa') is not None

    try:
        subprocess.run(['which', 'Xvfb'], check=True, capture_output=True)
        xvfb_available = True
    except:
        xvfb_available = False

    # Set the best backend before any MuJoCo imports
    if osmesa_available:
        print("Pre-setting OSMesa backend...")
        os.environ['MUJOCO_GL'] = 'osmesa'
        os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
        os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'
    elif xvfb_available:
        print("Pre-setting Xvfb backend...")
        # Will start Xvfb later, but set the GL mode now
        os.environ['MUJOCO_GL'] = 'glfw'
    else:
        print("Pre-setting EGL backend...")
        os.environ['MUJOCO_GL'] = 'egl'
        os.environ['PYOPENGL_PLATFORM'] = 'egl'

# Initialize BEFORE any dm_control imports
initialize_dmc_environment()

# NOW import dm_control after environment is set up
from dm_control import suite
from dm_control.suite.wrappers import pixels
from models.RSSM import RSSM
from collections import deque
import random
from torch.distributions import Normal, kl_divergence
import wandb
from cem_planner import PlaNetController

def check_rendering_libraries():
    """Check which rendering libraries are available"""
    print("=== Checking Rendering Libraries ===")

    # Check OSMesa
    try:
        import ctypes.util
        osmesa_lib = ctypes.util.find_library('OSMesa')
        if osmesa_lib:
            print(f"✓ OSMesa library found: {osmesa_lib}")
        else:
            print("✗ OSMesa library not found")
    except Exception as e:
        print(f"✗ Error checking OSMesa: {e}")

    # Check MuJoCo
    try:
        import mujoco
        print(f"✓ MuJoCo version: {mujoco.__version__}")
    except Exception as e:
        print(f"✗ Error importing MuJoCo: {e}")

    # Check PyOpenGL
    try:
        import OpenGL
        print(f"✓ PyOpenGL version: {OpenGL.__version__}")
    except Exception as e:
        print(f"✗ Error importing PyOpenGL: {e}")

    print("=" * 50)

def detect_best_rendering_backend():
    """Detect the best rendering backend for this environment"""
    import ctypes.util
    import subprocess

    print("=== Detecting Best Rendering Backend ===")

    # Check for OSMesa library
    osmesa_available = ctypes.util.find_library('OSMesa') is not None
    print(f"OSMesa library: {'✓ Available' if osmesa_available else '✗ Not found'}")

    # Check for Xvfb
    try:
        subprocess.run(['which', 'Xvfb'], check=True, capture_output=True)
        xvfb_available = True
        print("Xvfb: ✓ Available")
    except:
        xvfb_available = False
        print("Xvfb: ✗ Not found")

    # Choose backend based on availability
    if osmesa_available:
        print("→ Using OSMesa backend (software rendering)")
        return 'osmesa'
    elif xvfb_available:
        print("→ Using Xvfb + GLFW backend (virtual display)")
        return 'xvfb'
    else:
        print("→ Trying EGL backend (hardware rendering)")
        return 'egl'


def setup_rendering_environment(backend):
    """Setup environment variables for the chosen backend"""

    # Clear any existing conflicting variables
    vars_to_clear = ['DISPLAY', 'XAUTHORITY', 'MUJOCO_GL', 'PYOPENGL_PLATFORM',
                     'LIBGL_ALWAYS_SOFTWARE', 'MESA_GL_VERSION_OVERRIDE']

    for var in vars_to_clear:
        if var in os.environ:
            del os.environ[var]

    if backend == 'osmesa':
        print("Setting up OSMesa software rendering...")
        os.environ['MUJOCO_GL'] = 'osmesa'
        os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
        os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'

    elif backend == 'xvfb':
        print("Setting up Xvfb virtual display...")
        import subprocess
        import time

        # Start Xvfb
        subprocess.Popen([
            'Xvfb', ':99', '-screen', '0', '1024x768x24', '-ac', '+extension', 'GLX'
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        time.sleep(3)  # Give Xvfb time to start
        os.environ['DISPLAY'] = ':99'
        os.environ['MUJOCO_GL'] = 'glfw'

    elif backend == 'egl':
        print("Setting up EGL rendering...")
        os.environ['MUJOCO_GL'] = 'egl'
        os.environ['PYOPENGL_PLATFORM'] = 'egl'

    print(f"Environment variables set: MUJOCO_GL={os.environ.get('MUJOCO_GL')}")


def create_dmc_env_safe(domain_name="walker", task_name="walk", height=64, width=64, camera_id=0):
    """Create DMC environment with single-shot rendering setup"""

    check_rendering_libraries()

    # Detect and setup the best backend ONCE at startup
    backend = detect_best_rendering_backend()
    setup_rendering_environment(backend)

    print(f"\n=== Creating DMC Environment with {backend.upper()} backend ===")

    try:
        # Import AFTER setting up environment
        from dm_control import suite
        from dm_control.suite.wrappers import pixels

        print("Loading DMC suite environment...")
        env = suite.load(domain_name=domain_name, task_name=task_name)

        print("Adding pixel wrapper...")
        env = pixels.Wrapper(
            env,
            pixels_only=True,
            render_kwargs={'height': height, 'width': width, 'camera_id': camera_id}
        )

        print("Testing environment reset and rendering...")
        time_step = env.reset()
        pixels_obs = time_step.observation['pixels']
        print(f"✓ SUCCESS! Rendered observation shape: {pixels_obs.shape}")

        return env

    except Exception as e:
        print(f"\n✗ FAILED with {backend} backend")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

        print("\n" + "="*60)
        print("RENDERING SETUP FAILED")
        print("Please install missing packages:")
        print("  sudo apt update")
        print("  sudo apt install -y libosmesa6-dev mesa-utils xvfb")
        print("="*60)

        raise RuntimeError(f"DMC environment creation failed with {backend} backend")


class CPURenderWrapper:
    """Custom wrapper for CPU-only rendering"""

    def __init__(self, env, height=64, width=64, camera_id=0):
        self._env = env
        self._height = height
        self._width = width
        self._camera_id = camera_id

    def reset(self):
        time_step = self._env.reset()
        # Add CPU-rendered pixels to observation
        pixels = self._env.physics.render(
            height=self._height,
            width=self._width,
            camera_id=self._camera_id
        )

        # Create observation dict similar to pixels wrapper
        obs = {'pixels': pixels}

        # Return modified time_step
        from dm_control.rl import control
        return control.TimeStep(
            step_type=time_step.step_type,
            reward=time_step.reward,
            discount=time_step.discount,
            observation=obs
        )

    def step(self, action):
        time_step = self._env.step(action)
        # Add CPU-rendered pixels to observation
        pixels = self._env.physics.render(
            height=self._height,
            width=self._width,
            camera_id=self._camera_id
        )

        # Create observation dict similar to pixels wrapper
        obs = {'pixels': pixels}

        # Return modified time_step
        from dm_control.rl import control
        return control.TimeStep(
            step_type=time_step.step_type,
            reward=time_step.reward,
            discount=time_step.discount,
            observation=obs
        )

    def action_spec(self):
        return self._env.action_spec()

    def observation_spec(self):
        # Mock the pixels observation spec
        return {'pixels': self._env.observation_spec()}

    def close(self):
        self._env.close()

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
        """Sample sequences that don't cross episode boundaries"""
        if len(self.observations) < sequence_length:
            return None, None, None
        
        # Build list of valid (start_idx, end_idx) ranges for each episode
        valid_starts = []
        current_idx = 0
        
        for ep_len in self.episode_lengths:
            if ep_len >= sequence_length:
                # Can sample any start position where we have seq_len steps remaining
                for start in range(current_idx, current_idx + ep_len - sequence_length + 1):
                    valid_starts.append(start)
            current_idx += ep_len
        
        if len(valid_starts) < batch_size:
            # Not enough valid starting points - sample with replacement
            if len(valid_starts) == 0:
                return None, None, None
            chosen_starts = random.choices(valid_starts, k=batch_size)
        else:
            chosen_starts = random.sample(valid_starts, batch_size)
        
        obs_batch = []
        action_batch = []
        reward_batch = []
        
        for start_idx in chosen_starts:
            obs_seq = self.observations[start_idx:start_idx + sequence_length]
            action_seq = self.actions[start_idx:start_idx + sequence_length]
            reward_seq = self.rewards[start_idx:start_idx + sequence_length]
            
            obs_batch.append(obs_seq)
            action_batch.append(action_seq)
            reward_batch.append(reward_seq)
        
        return (
            torch.stack([torch.stack(seq) for seq in obs_batch]),
            torch.stack([torch.stack(seq) for seq in action_batch]),
            torch.stack([torch.stack(seq) for seq in reward_batch])
        )

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

    print(f"\n{'='*60}")
    print(f"🎯 STARTING RANDOM EPISODE COLLECTION")
    print(f"Target episodes: {num_episodes}")
    print(f"Max steps per episode: {max_steps_per_episode}")
    print(f"{'='*60}")

    total_steps = 0
    total_reward = 0.0

    for episode in range(num_episodes):
        print(f"\n📺 Episode {episode + 1}/{num_episodes} - Starting...")

        obs_sequence = []
        action_sequence = []
        reward_sequence = []
        episode_reward = 0.0
        episode_steps = 0

        # Reset environment
        print(f"  🔄 Resetting environment...")
        obs, info = env.reset()
        # Convert image from (H, W, C) to (C, H, W) and normalize to [0, 1]
        # Use .copy() to ensure contiguous array for PyTorch
        obs_tensor = torch.tensor(obs.copy(), dtype=torch.float32).permute(2, 0, 1) / 255.0
        obs_sequence.append(obs_tensor)
        print(f"  ✅ Environment reset complete. Observation shape: {obs.shape}")

        # Episode rollout
        for step in range(max_steps_per_episode):
            # Random action sampling
            action = env.action_space.sample()
            action_tensor = torch.tensor(action, dtype=torch.float32)
            action_sequence.append(action_tensor)

            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            # Convert image from (H, W, C) to (C, H, W) and normalize to [0, 1]
            # Use .copy() to ensure contiguous array for PyTorch
            obs_tensor = torch.tensor(obs.copy(), dtype=torch.float32).permute(2, 0, 1) / 255.0
            obs_sequence.append(obs_tensor)
            reward_sequence.append(torch.tensor(reward, dtype=torch.float32))

            episode_reward += reward
            episode_steps += 1

            # Progress indicator every 100 steps
            if (step + 1) % 100 == 0:
                print(f"    Step {step + 1}/{max_steps_per_episode} | Reward: {episode_reward:.2f}")

            if terminated or truncated:
                reason = "terminated" if terminated else "truncated"
                print(f"  🏁 Episode ended at step {episode_steps} ({reason})")
                break

        # Add episode to buffer (excluding last observation for action alignment)
        buffer.add_episode(obs_sequence[:-1], action_sequence, reward_sequence)

        total_steps += episode_steps
        total_reward += episode_reward
        avg_reward = total_reward / (episode + 1)
        avg_steps = total_steps / (episode + 1)

        print(f"  ✅ Episode {episode + 1} complete!")
        print(f"     Steps: {episode_steps} | Reward: {episode_reward:.3f}")
        print(f"     Running averages - Steps: {avg_steps:.1f} | Reward: {avg_reward:.3f}")

        # Milestone updates
        if (episode + 1) % 5 == 0:
            print(f"\n📊 PROGRESS UPDATE:")
            print(f"   Episodes completed: {episode + 1}/{num_episodes}")
            print(f"   Total timesteps collected: {len(buffer.observations)}")
            print(f"   Average episode length: {avg_steps:.1f} steps")
            print(f"   Average episode reward: {avg_reward:.3f}")

    print(f"\n{'='*60}")
    print(f"🎉 RANDOM EPISODE COLLECTION COMPLETE!")
    print(f"Total episodes: {num_episodes}")
    print(f"Total timesteps: {len(buffer.observations)}")
    print(f"Average episode length: {total_steps / num_episodes:.1f} steps")
    print(f"Average episode reward: {total_reward / num_episodes:.3f}")
    print(f"{'='*60}")

    return buffer

def compute_losses(rssm_output, reconstructed_obs, target_obs, predicted_rewards, target_rewards, 
                   free_nats=3.0, debug=False):
    """
    Compute RSSM training losses with summed pixel log-probs.
    """
    prior_states, posterior_states, hiddens, prior_mus, prior_stds, \
        posterior_mus, posterior_stds, rewards = rssm_output

    obs_dist = Normal(reconstructed_obs, 1.0)
    full_log_prob = obs_dist.log_prob(target_obs)
    reconstruction_loss = -full_log_prob.sum(dim=(2, 3, 4)).mean()

    reward_dist = Normal(predicted_rewards, 1.0)
    if target_rewards.dim() == 2:
        target_rewards = target_rewards.unsqueeze(-1)
    reward_loss = -reward_dist.log_prob(target_rewards).sum(dim=-1).mean()

    prior_dist = Normal(prior_mus, prior_stds)
    posterior_dist = Normal(posterior_mus, posterior_stds)
    
    kl_per_timestep = kl_divergence(posterior_dist, prior_dist).sum(dim=-1)
    raw_kl = kl_per_timestep.mean()

    free_nats_tensor = torch.tensor(free_nats, device=kl_per_timestep.device)
    kl_loss = torch.max(kl_per_timestep, free_nats_tensor).mean()
    
    if debug:
        print(f"    DEBUG - Recon Loss (Summed): {reconstruction_loss.item():.2f}")
        print(f"    DEBUG - Reward Loss: {reward_loss.item():.2f}")
        print(f"    DEBUG - Raw KL: {raw_kl.item():.2f}")

    return reconstruction_loss, reward_loss, kl_loss, raw_kl


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
        # Convert from [H, W, C] to [C, H, W] for PyTorch CNN and normalize
        obs_tensor = torch.tensor(obs.copy(), dtype=torch.float32).permute(2, 0, 1) / 255.0

        controller.reset(obs_tensor)
        episode_return = 0.0

        for step in range(max_steps):
            # Get action from CEM planner
            action = controller.act(obs_tensor)

            # Take action in environment
            obs, reward, terminated, truncated, info = env.step(action)
            # Convert from [H, W, C] to [C, H, W] for PyTorch CNN and normalize
            obs_tensor = torch.tensor(obs.copy(), dtype=torch.float32).permute(2, 0, 1) / 255.0

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

def collect_cem_episodes(rssm, env, num_episodes=5, max_steps=1000, action_repeat=2, exploration_noise=0.3):
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

    print(f"\n{'='*60}")
    print(f"🤖 STARTING CEM PLANNING EPISODE COLLECTION")
    print(f"Target episodes: {num_episodes}")
    print(f"Max steps per episode: {max_steps}")
    print(f"Action repeat (R): {action_repeat}")
    print(f"Planning horizon: 12")
    print(f"{'='*60}")

    total_steps = 0
    total_reward = 0.0

    for episode in range(num_episodes):
        print(f"\n🎯 CEM Episode {episode + 1}/{num_episodes} - Starting...")

        obs_sequence = []
        action_sequence = []
        reward_sequence = []
        episode_reward = 0.0
        episode_steps = 0

        # Reset environment and controller
        print(f"  🔄 Resetting environment and CEM controller...")
        obs, info = env.reset()
        obs_tensor = torch.tensor(obs.copy(), dtype=torch.float32).permute(2, 0, 1) / 255.0
        obs_sequence.append(obs_tensor)

        controller.reset(obs_tensor)
        print(f"  ✅ Reset complete. Starting CEM-guided rollout...")

        for step in range(max_steps):
            # Get action from CEM planner (every R timesteps)
            if step % 10 == 0:  # More frequent progress updates every 10 steps
                print(f"    🧠 Step {step}/{max_steps} | Planning action...")

            # Add timing to see if CEM planning is slow
            import time
            start_time = time.time()
            action = controller.act(obs_tensor)
            noise = np.random.normal(0, exploration_noise, size=action.shape)
            action = action + noise
            action = np.clip(action, -1.0, 1.0)  # Clip to valid range

            action_tensor = torch.tensor(action, dtype=torch.float32)
            action_sequence.append(action_tensor)

            # Take action in environment
            obs, reward, terminated, truncated, info = env.step(action)
            obs_tensor = torch.tensor(obs.copy(), dtype=torch.float32).permute(2, 0, 1) / 255.0
            obs_sequence.append(obs_tensor)
            reward_sequence.append(torch.tensor(reward, dtype=torch.float32))

            episode_reward += reward
            episode_steps += 1

            # Update controller's internal state
            controller.update_state(action)

            # Progress indicator every 100 steps
            if (step + 1) % 100 == 0:
                print(f"    Step {step + 1}/{max_steps} | Reward: {episode_reward:.2f}")

            if terminated or truncated:
                reason = "terminated" if terminated else "truncated"
                print(f"  🏁 CEM episode ended at step {episode_steps} ({reason})")
                break

        # Add episode to buffer (excluding last observation for action alignment)
        buffer.add_episode(obs_sequence[:-1], action_sequence, reward_sequence)

        total_steps += episode_steps
        total_reward += episode_reward
        avg_reward = total_reward / (episode + 1)
        avg_steps = total_steps / (episode + 1)

        print(f"  ✅ CEM Episode {episode + 1} complete!")
        print(f"     Steps: {episode_steps} | Reward: {episode_reward:.3f}")
        print(f"     Running averages - Steps: {avg_steps:.1f} | Reward: {avg_reward:.3f}")

        # Progress update
        if (episode + 1) % 2 == 0:  # More frequent updates for CEM (every 2 episodes)
            print(f"\n📊 CEM PROGRESS UPDATE:")
            print(f"   CEM episodes completed: {episode + 1}/{num_episodes}")
            print(f"   Total CEM timesteps collected: {len(buffer.observations)}")
            print(f"   Average CEM episode length: {avg_steps:.1f} steps")
            print(f"   Average CEM episode reward: {avg_reward:.3f}")

    print(f"\n{'='*60}")
    print(f"🎉 CEM EPISODE COLLECTION COMPLETE!")
    print(f"Total CEM episodes: {num_episodes}")
    print(f"Total CEM timesteps: {len(buffer.observations)}")
    print(f"Average CEM episode length: {total_steps / num_episodes:.1f} steps")
    print(f"Average CEM episode reward: {total_reward / num_episodes:.3f}")
    print(f"{'='*60}")

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

    # Initialize DMC walker environment with pixel observations
    print("Initializing DMC walker environment...")
    dmc_env = create_dmc_env_safe(domain_name="walker", task_name="walk", height=64, width=64, camera_id=0)
    env = DMCWrapper(dmc_env)
    environment_name = "DMC-walker-walk"

    # Environment specifications
    obs_shape = env.observation_space.shape 
    action_dim = env.action_space.shape[0]

    # RSSM hyperparameters
    encoded_size = 1024
    latent_size = 30
    hidden_size = 200

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
            "obs_shape": obs_shape,
            "action_dim": action_dim,
            "plan_every": plan_every,
            "planning_episodes": planning_episodes,
            "action_repeat": action_repeat,
            "environment": environment_name
        }
    )

    # Setup device (GPU if available, CPU otherwise)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA Version: {torch.version.cuda}")

    # Initialize RSSM
    rssm = RSSM(
        action_size=action_dim,
        latent_size=latent_size,
        encoded_size=encoded_size,
        hidden_size=hidden_size,
        min_std_dev=0.1,
        device=device
    ).to(device)
    optimizer = optim.Adam(rssm.parameters(), lr=learning_rate, eps=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

    # Collect initial dataset
    dataset = collect_random_episodes(env, S)

    # Log dataset statistics
    wandb.log({
        "dataset_size": len(dataset.observations),
        "num_episodes": S,
        "avg_episode_length": len(dataset.observations) / S if S > 0 else 0
    })

    # Training loop
    best_eval_return = float('-inf')
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(num_epochs):
        if epoch % 10 == 0:
            print(f"\n🔄 Starting epoch {epoch}/{num_epochs}...")
            print(f"   Current dataset size: {len(dataset.observations)} timesteps")
        # Sample batch of sequences
        obs_batch, action_batch, reward_batch = dataset.get_random_sequences(B, L)

        if obs_batch is None:
            print("Not enough data for training batch, skipping...")
            continue

        # Move data to device
        obs_batch = obs_batch.to(device)
        action_batch = action_batch.to(device)
        reward_batch = reward_batch.to(device)

        # Debug tensor shapes
        if epoch % 10 == 0:
            print(f"   Batch shapes - Obs: {obs_batch.shape}, Actions: {action_batch.shape}, Rewards: {reward_batch.shape}")

        optimizer.zero_grad()

        # Encode observations
        # obs_batch shape: [batch_size, seq_len, channels, height, width]
        batch_size, seq_len = obs_batch.shape[:2]
        obs_channels, obs_height, obs_width = obs_batch.shape[2:]

        # Only print batch info on milestone epochs to avoid spam
        if epoch % 10 == 0:
            print(f"   Processing batch: {batch_size} sequences x {seq_len} timesteps")
            print(f"   Observation shape: {obs_channels}x{obs_height}x{obs_width}")

        # Flatten for encoding: [batch_size * seq_len, channels, height, width]
        flat_obs = obs_batch.view(-1, obs_channels, obs_height, obs_width)
        encoded_obs = rssm.encode(flat_obs)

        # Reshape back: [batch_size, seq_len, encoded_dim]
        encoded_dim = encoded_obs.shape[-1]
        encoded_obs = encoded_obs.view(batch_size, seq_len, encoded_dim)

        encoded_obs_for_posterior = encoded_obs[:, 1:, :]      # o_1 to o_{L-1} (next observations)
        obs_targets = obs_batch[:, 1:, :, :, :]                # Raw pixels for reconstruction loss
        action_batch_aligned = action_batch[:, :-1, :]         # a_0 to a_{L-2}
        reward_batch_aligned = reward_batch[:, :-1]            # r_0 to r_{L-2}

        effective_seq_len = seq_len - 1

        # Debug encoded dimensions on first epoch
        if epoch == 0:
            print(f"   Encoded observation shape: [batch_size, seq_len, encoded_dim] = {encoded_obs.shape}")

        # Initialize states
        prev_state = torch.zeros(batch_size, latent_size, device=device)
        prev_hidden = torch.zeros(batch_size, hidden_size, device=device)

        rssm_output = rssm.pass_through(
            prev_state, 
            prev_hidden, 
            encoded_obs_for_posterior,
            action_batch_aligned
        )

        prior_states, posterior_states, hiddens, _, _, _, _, predicted_rewards = rssm_output
        reconstructed = []

        for t in range(effective_seq_len):
            decoded = rssm.decode(hiddens[:, t, :], posterior_states[:, t, :])
            reconstructed.append(decoded)

        reconstructed_obs = torch.stack(reconstructed, dim=1)

        reconstruction_loss, reward_loss, kl_loss, raw_kl = compute_losses(
            rssm_output, 
            reconstructed_obs, 
            obs_targets,
            predicted_rewards, 
            reward_batch_aligned,
            free_nats=3.0,
            debug=(epoch % 50 == 0)
        )

        total_loss = reconstruction_loss + 10 * reward_loss + kl_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(rssm.parameters(), max_norm=1000.0)  # Paper uses 1000
        optimizer.step()
        scheduler.step()

        wandb.log({
            "epoch": epoch,
            "total_loss": total_loss.item(),
            "reconstruction_loss": reconstruction_loss.item(),
            "reward_loss": reward_loss.item(),
            "kl_loss": kl_loss.item(),
            "raw_kl": raw_kl.item(),  # Add this line
            "learning_rate": scheduler.get_last_lr()[0]
        })

        if epoch % 10 == 0:
            print(f"📈 Epoch {epoch}: Total Loss: {total_loss.item():.4f}, "
                  f"Reconstruction: {reconstruction_loss.item():.4f}, "
                  f"Reward: {reward_loss.item():.4f}, "
                  f"KL: {kl_loss.item():.4f}")

        # Quick progress indicator for non-milestone epochs
        elif epoch % 5 == 0:
            print(f"   Epoch {epoch}: Loss={total_loss.item():.4f}")

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

            # Save checkpoint if this is the best performance so far
            if avg_return > best_eval_return:
                best_eval_return = avg_return
                best_checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': rssm.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_eval_return': best_eval_return,
                    'avg_return': avg_return,
                    'episode_returns': episode_returns
                }, best_checkpoint_path)
                print(f"🎯 New best model saved! Return: {avg_return:.2f}")

            # Save periodic checkpoint
            if epoch % (evaluate_every * 2) == 0:  # Every 2nd evaluation
                periodic_checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': rssm.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_eval_return': best_eval_return,
                    'avg_return': avg_return,
                    'episode_returns': episode_returns
                }, periodic_checkpoint_path)
                print(f"📁 Periodic checkpoint saved: epoch_{epoch}.pth")

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
        S=5, B=50, L=50,
        num_epochs= 50000,
        evaluate_every=500,
        evaluation_episodes=1,
        plan_every=100,  # CEM planning every 25 epochs
        planning_episodes=1,
        action_repeat=2  # Action repeat parameter R
    )

    print("Training complete! Model saved as 'trained_rssm_dmc_walker.pth'")

    # Final evaluation using DMC walker
    print("\n=== Final Evaluation ===")
    final_dmc_env = create_dmc_env_safe(domain_name="walker", task_name="walk", height=64, width=64, camera_id=0)
    final_eval_env = DMCWrapper(final_dmc_env)

    final_avg_return, _ = evaluate_controller(trained_rssm, final_eval_env, num_episodes=10)
    final_eval_env.close()
    print(f"Final Average Return: {final_avg_return:.2f}")