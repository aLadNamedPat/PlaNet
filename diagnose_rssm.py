import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Set up rendering BEFORE any dm_control imports
os.environ['MUJOCO_GL'] = 'osmesa'
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'

# Add the parent directory to path to import your modules
sys.path.insert(0, '/home/grokveritas/PlaNet')

from models.RSSM import RSSM
from dm_control import suite
from dm_control.suite.wrappers import pixels

def create_walker_env():
    """Create Walker environment with pixel observations"""
    env = suite.load(domain_name="walker", task_name="walk")
    env = pixels.Wrapper(
        env,
        pixels_only=True,
        render_kwargs={'height': 64, 'width': 64, 'camera_id': 0}
    )
    return env

def collect_observations(env, num_obs=10, num_steps_between=10):
    """Collect real observations from Walker environment"""
    observations = []
    
    time_step = env.reset()
    obs = time_step.observation['pixels']
    obs_tensor = torch.tensor(obs.copy(), dtype=torch.float32).permute(2, 0, 1) / 255.0
    observations.append(obs_tensor)
    
    action_spec = env.action_spec()
    
    for i in range(num_obs - 1):
        # Take several random steps to get diverse observations
        for _ in range(num_steps_between):
            action = np.random.uniform(action_spec.minimum, action_spec.maximum)
            time_step = env.step(action)
        
        obs = time_step.observation['pixels']
        obs_tensor = torch.tensor(obs.copy(), dtype=torch.float32).permute(2, 0, 1) / 255.0
        observations.append(obs_tensor)
        
        # Reset if episode ended
        if time_step.last():
            time_step = env.reset()
    
    return observations

def load_rssm(checkpoint_path, device='cpu'):
    """Load RSSM from checkpoint"""
    # RSSM hyperparameters (must match training)
    action_dim = 6  # Walker has 6 action dims
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
    
    return rssm

def diagnose_rssm(checkpoint_path, device='cpu'):
    """Run comprehensive diagnostics on RSSM"""
    
    print(f"\n{'='*60}")
    print(f"RSSM DIAGNOSTIC REPORT")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"{'='*60}\n")
    
    # Create Walker environment and collect real observations
    print("Creating Walker environment...")
    env = create_walker_env()
    
    print("Collecting real observations from Walker...")
    observations = collect_observations(env, num_obs=10, num_steps_between=20)
    print(f"Collected {len(observations)} observations")
    
    rssm = load_rssm(checkpoint_path, device)
    
    # Use real observations
    test_obs = observations[0].unsqueeze(0).to(device)  # First observation
    test_obs_2 = observations[5].unsqueeze(0).to(device)  # Different observation
    
    with torch.no_grad():
        # ============================================================
        # TEST 1: Encoder Output
        # ============================================================
        print("\n" + "=" * 40)
        print("TEST 1: ENCODER OUTPUT")
        print("=" * 40)
        
        encoded_1 = rssm.encode(test_obs)
        encoded_2 = rssm.encode(test_obs_2)
        
        print(f"Input obs shape: {test_obs.shape}")
        print(f"Encoded shape: {encoded_1.shape}")
        print(f"\nEncoded obs 1 stats:")
        print(f"  Mean: {encoded_1.mean().item():.4f}")
        print(f"  Std:  {encoded_1.std().item():.4f}")
        print(f"  Min:  {encoded_1.min().item():.4f}")
        print(f"  Max:  {encoded_1.max().item():.4f}")
        
        print(f"\nEncoded obs 2 stats:")
        print(f"  Mean: {encoded_2.mean().item():.4f}")
        print(f"  Std:  {encoded_2.std().item():.4f}")
        print(f"  Min:  {encoded_2.min().item():.4f}")
        print(f"  Max:  {encoded_2.max().item():.4f}")
        
        encoding_diff = (encoded_1 - encoded_2).abs().mean().item()
        print(f"\nDifference between encodings: {encoding_diff:.4f}")
        
        if encoded_1.std().item() < 0.01:
            print("⚠️  WARNING: Encoder output has very low variance!")
        if encoding_diff < 0.01:
            print("⚠️  WARNING: Different inputs produce nearly identical encodings!")
        
        # ============================================================
        # TEST 2: Prior vs Posterior
        # ============================================================
        print("\n" + "=" * 40)
        print("TEST 2: PRIOR vs POSTERIOR")
        print("=" * 40)
        
        # Initialize hidden state
        hidden = torch.zeros(1, 200, device=device)
        
        # Sample from prior (no observation)
        prior_sample, prior_mu, prior_std = rssm.sample_prior(hidden)
        
        # Sample from posterior (with observation)
        posterior_sample, posterior_mu, posterior_std = rssm.sample_posterior(hidden, encoded_1)
        
        print(f"\nPrior stats:")
        print(f"  Mu mean:  {prior_mu.mean().item():.4f}")
        print(f"  Mu std:   {prior_mu.std().item():.4f}")
        print(f"  Sigma mean: {prior_std.mean().item():.4f}")
        
        print(f"\nPosterior stats (with obs 1):")
        print(f"  Mu mean:  {posterior_mu.mean().item():.4f}")
        print(f"  Mu std:   {posterior_mu.std().item():.4f}")
        print(f"  Sigma mean: {posterior_std.mean().item():.4f}")
        
        mu_diff = (posterior_mu - prior_mu).abs().mean().item()
        print(f"\n|Posterior_mu - Prior_mu| mean: {mu_diff:.4f}")
        
        if mu_diff < 0.01:
            print("🚨 PROBLEM: Posterior is nearly identical to prior!")
            print("   The model is ignoring observations!")
        
        # Test with second observation
        posterior_sample_2, posterior_mu_2, posterior_std_2 = rssm.sample_posterior(hidden, encoded_2)
        
        posterior_diff = (posterior_mu - posterior_mu_2).abs().mean().item()
        print(f"\nPosterior difference for different observations: {posterior_diff:.4f}")
        
        if posterior_diff < 0.01:
            print("🚨 PROBLEM: Different observations produce same posterior!")
        
        # ============================================================
        # TEST 3: Decoder Reconstruction
        # ============================================================
        print("\n" + "=" * 40)
        print("TEST 3: DECODER RECONSTRUCTION")
        print("=" * 40)
        
        # Decode from posterior state
        reconstructed = rssm.decode(hidden, posterior_sample)
        
        print(f"Reconstructed shape: {reconstructed.shape}")
        print(f"Reconstructed stats:")
        print(f"  Mean: {reconstructed.mean().item():.4f}")
        print(f"  Std:  {reconstructed.std().item():.4f}")
        print(f"  Min:  {reconstructed.min().item():.4f}")
        print(f"  Max:  {reconstructed.max().item():.4f}")
        
        recon_error = ((reconstructed - test_obs) ** 2).mean().item()
        print(f"\nReconstruction MSE: {recon_error:.4f}")
        
        # ============================================================
        # TEST 4: Full Pass Through (Sequence)
        # ============================================================
        print("\n" + "=" * 40)
        print("TEST 4: FULL PASS THROUGH (5 timesteps)")
        print("=" * 40)
        
        batch_size = 1
        seq_len = 5
        action_dim = 6
        latent_size = 30
        hidden_size = 200
        
        # Use real observations for sequence
        obs_seq = torch.stack(observations[:seq_len]).unsqueeze(0).to(device)  # [1, 5, 3, 64, 64]
        action_seq = torch.rand(batch_size, seq_len, action_dim, device=device) * 2 - 1  # [-1, 1]
        
        # Encode all observations
        flat_obs = obs_seq.view(-1, 3, 64, 64)
        encoded_seq = rssm.encode(flat_obs).view(batch_size, seq_len, -1)
        
        # Initialize states
        prev_state = torch.zeros(batch_size, latent_size, device=device)
        prev_hidden = torch.zeros(batch_size, hidden_size, device=device)
        
        # Run pass_through
        output = rssm.pass_through(prev_state, prev_hidden, encoded_seq, action_seq)
        prior_states, posterior_states, hiddens, prior_mus, prior_stds, posterior_mus, posterior_stds, rewards = output
        
        print(f"Output shapes:")
        print(f"  Prior states: {prior_states.shape}")
        print(f"  Posterior states: {posterior_states.shape}")
        print(f"  Hiddens: {hiddens.shape}")
        print(f"  Rewards: {rewards.shape}")
        
        # Check KL divergence manually
        from torch.distributions import Normal, kl_divergence
        
        prior_dist = Normal(prior_mus, prior_stds)
        posterior_dist = Normal(posterior_mus, posterior_stds)
        kl = kl_divergence(posterior_dist, prior_dist).sum(dim=-1).mean()
        
        print(f"\nKL divergence (summed over latent dims): {kl.item():.4f}")
        
        # Per-timestep analysis
        print(f"\nPer-timestep |posterior_mu - prior_mu|:")
        for t in range(seq_len):
            diff = (posterior_mus[:, t, :] - prior_mus[:, t, :]).abs().mean().item()
            print(f"  t={t}: {diff:.4f}")
        
        # ============================================================
        # TEST 5: Visualize Reconstruction with REAL observations
        # ============================================================
        print("\n" + "=" * 40)
        print("TEST 5: SAVING VISUALIZATION (Real Walker Observations)")
        print("=" * 40)
        
        # Create a figure with original and reconstructed images
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        
        # Use 5 different real observations
        for i in range(5):
            obs = observations[i].unsqueeze(0).to(device)
            encoded = rssm.encode(obs)
            hidden = torch.zeros(1, 200, device=device)
            posterior_sample, _, _ = rssm.sample_posterior(hidden, encoded)
            reconstructed = rssm.decode(hidden, posterior_sample)
            
            # Original
            orig_img = obs[0].permute(1, 2, 0).cpu().numpy()
            orig_img = np.clip(orig_img, 0, 1)
            axes[0, i].imshow(orig_img)
            axes[0, i].set_title(f'Original {i+1}')
            axes[0, i].axis('off')
            
            # Reconstructed
            recon_img = reconstructed[0].permute(1, 2, 0).cpu().numpy()
            recon_img = np.clip(recon_img, 0, 1)
            axes[1, i].imshow(recon_img)
            mse = ((reconstructed[0] - obs[0]) ** 2).mean().item()
            axes[1, i].set_title(f'Recon {i+1}\nMSE: {mse:.4f}')
            axes[1, i].axis('off')
        
        plt.suptitle(f'RSSM Reconstruction Test (Real Walker Obs)\n{os.path.basename(checkpoint_path)}')
        plt.tight_layout()
        
        output_path = 'rssm_diagnostic.png'
        plt.savefig(output_path, dpi=150)
        print(f"Saved visualization to: {output_path}")
        
        # ============================================================
        # SUMMARY
        # ============================================================
        print("\n" + "=" * 60)
        print("DIAGNOSTIC SUMMARY")
        print("=" * 60)
        
        issues = []
        
        if encoded_1.std().item() < 0.1:
            issues.append("Encoder has low variance output")
        
        if mu_diff < 0.1:
            issues.append("Posterior ≈ Prior (ignoring observations)")
        
        if posterior_diff < 0.1:
            issues.append("Different observations → same posterior")
        
        if kl.item() < 1.0:
            issues.append(f"KL divergence very low ({kl.item():.2f} nats)")
        
        if len(issues) == 0:
            print("✅ No major issues detected!")
        else:
            print("🚨 Issues found:")
            for issue in issues:
                print(f"   - {issue}")
        
        env.close()
        return rssm

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth',
                        help='Path to checkpoint file')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    diagnose_rssm(args.checkpoint, args.device)