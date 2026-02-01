import torch
import torch.nn as nn
import numpy as np

class CEMPlanner:
    """Cross-Entropy Method planner for PlaNet - Ultra-Fast Version"""

    def __init__(self, rssm, action_dim, horizon=12, iterations=6, candidates=50, top_k=5, device='cpu'):
        """
        Initialize CEM planner

        Args:
            rssm: Trained RSSM model
            action_dim: Dimensionality of action space
            horizon: Planning horizon H
            iterations: Number of optimization iterations I
            candidates: Number of candidate sequences J
            top_k: Number of top candidates K to refit belief
            device: torch device
        """
        self.rssm = rssm
        self.action_dim = action_dim
        self.horizon = horizon
        self.iterations = iterations
        self.candidates = candidates
        self.top_k = top_k
        self.device = device

        # Action bounds (assuming [-1, 1] for continuous control)
        self.action_min = -1.0
        self.action_max = 1.0

    def plan(self, current_state_belief, current_hidden):
        """
        Plan optimal action using CEM

        Args:
            current_state_belief: Current state belief q(s_t | o_≤t, a_<t) [batch_size, latent_size]
            current_hidden: Current hidden state [batch_size, hidden_size]

        Returns:
            action: Optimal action for current timestep [action_dim]
        """
        batch_size = current_state_belief.shape[0]

        # Initialize factorized belief over action sequences: Normal(0, I)
        mean = torch.zeros(self.horizon, self.action_dim, device=self.device)
        std = torch.ones(self.horizon, self.action_dim, device=self.device)

        # Add debug print for first few planning calls
        if not hasattr(self, '_plan_call_count'):
            self._plan_call_count = 0

        if self._plan_call_count < 3:
            print(f"      CEM Plan call #{self._plan_call_count} - State: {current_state_belief.shape}, Hidden: {current_hidden.shape}")
            print(f"      CEM Config: horizon={self.horizon}, iterations={self.iterations}, candidates={self.candidates}")
        self._plan_call_count += 1

        for iteration in range(self.iterations):
            if self._plan_call_count <= 3 and iteration == 0:
                print(f"      Starting CEM iteration {iteration}/{self.iterations}...")
            # Sample J candidate action sequences
            action_sequences = self._sample_action_sequences(mean, std, batch_size)

            # Evaluate each candidate sequence under the model
            returns = self._evaluate_sequences(action_sequences, current_state_belief, current_hidden)

            # Select top K sequences
            top_k_indices = torch.topk(returns, self.top_k, dim=0)[1]
            top_k_sequences = action_sequences[top_k_indices]

            # Refit belief to top K sequences
            mean, std = self._refit_belief(top_k_sequences)

        # Return mean action for current timestep
        return mean[0].cpu().numpy()  # First action in the sequence

    def _sample_action_sequences(self, mean, std, batch_size):
        """
        Sample candidate action sequences from current belief

        Args:
            mean: Current mean of belief [horizon, action_dim]
            std: Current std of belief [horizon, action_dim]
            batch_size: Batch size for state belief

        Returns:
            action_sequences: Sampled sequences [candidates, horizon, action_dim]
        """
        # Sample from diagonal Gaussian
        noise = torch.randn(self.candidates, self.horizon, self.action_dim, device=self.device)
        action_sequences = mean.unsqueeze(0) + std.unsqueeze(0) * noise

        # Clip to action bounds
        action_sequences = torch.clamp(action_sequences, self.action_min, self.action_max)

        return action_sequences

    def _evaluate_sequences(self, action_sequences, current_state_belief, current_hidden):
        """
        Evaluate action sequences under the learned model - VECTORIZED

        Args:
            action_sequences: [candidates, horizon, action_dim]
            current_state_belief: [batch_size, latent_size]
            current_hidden: [batch_size, hidden_size]

        Returns:
            returns: Expected returns for each sequence [candidates]
        """
        candidates = action_sequences.shape[0]
        batch_size = current_state_belief.shape[0]

        # Expand state belief and hidden state for all candidates
        # We'll evaluate each candidate independently (single trajectory per sequence)
        expanded_state = current_state_belief.unsqueeze(0).expand(candidates, -1, -1).reshape(-1, current_state_belief.shape[-1])
        expanded_hidden = current_hidden.unsqueeze(0).expand(candidates, -1, -1).reshape(-1, current_hidden.shape[-1])

        # Flatten action sequences for batch processing
        flat_actions = action_sequences.reshape(-1, self.horizon, self.action_dim)

        returns = []

        for i in range(candidates):
            # Get action sequence for this candidate
            candidate_actions = flat_actions[i:i+1]  # [1, horizon, action_dim]
            candidate_state = expanded_state[i*batch_size:(i+1)*batch_size]  # [batch_size, latent_size]
            candidate_hidden = expanded_hidden[i*batch_size:(i+1)*batch_size]  # [batch_size, hidden_size]

            # Rollout trajectory using the model (prior only for planning)
            trajectory_return = self._rollout_trajectory(candidate_actions, candidate_state, candidate_hidden)
            returns.append(trajectory_return)

        return torch.stack(returns)

    def _rollout_trajectory(self, action_sequence, initial_state, initial_hidden):
        """
        Rollout a single trajectory and compute return

        Args:
            action_sequence: [1, horizon, action_dim]
            initial_state: [batch_size, latent_size]
            initial_hidden: [batch_size, hidden_size]

        Returns:
            total_return: Sum of predicted rewards along trajectory
        """
        current_state = initial_state
        current_hidden = initial_hidden
        total_return = 0.0

        with torch.no_grad():  # No gradients needed for planning
            for t in range(self.horizon):
                action_t = action_sequence[:, t, :].expand(current_state.shape[0], -1)

                # Predict next state using prior (no observations available)
                state_action_embedding = self.rssm.state_action(
                    torch.cat([current_state, action_t], dim=-1)
                )

                # Update hidden state
                rnn_input = torch.cat([state_action_embedding, current_hidden], dim=-1).unsqueeze(1)
                _, current_hidden = self.rssm.rnn(rnn_input)
                current_hidden = current_hidden.squeeze(0)

                # Sample from prior distribution (deterministic for planning)
                current_state, _, _ = self.rssm.sample_prior(current_hidden, deterministic=True)

                # Predict reward
                predicted_reward = self.rssm.reward(
                    torch.cat([current_state, current_hidden], dim=-1)
                )

                total_return += predicted_reward.mean()  # Average over batch dimension

        return total_return

    def _rollout_trajectory_vectorized(self, action_sequences, initial_states, initial_hiddens):
        """
        Vectorized rollout for multiple trajectories simultaneously

        Args:
            action_sequences: [candidates, horizon, action_dim]
            initial_states: [candidates, latent_size]
            initial_hiddens: [candidates, hidden_size]

        Returns:
            total_returns: [candidates] - returns for each sequence
        """
        candidates = action_sequences.shape[0]
        current_states = initial_states  # [candidates, latent_size]
        current_hiddens = initial_hiddens  # [candidates, hidden_size]
        total_returns = torch.zeros(candidates, device=self.device)

        with torch.no_grad():
            for t in range(self.horizon):
                # Get actions for all candidates at timestep t
                actions_t = action_sequences[:, t, :].contiguous()  # [candidates, action_dim]

                # Compute state-action embeddings for all candidates
                # Create copies of tensors to avoid any reference issues
                current_states_copy = current_states.clone()
                actions_t_copy = actions_t.clone()

                state_action_input = torch.cat([current_states_copy, actions_t_copy], dim=-1)
                state_action_embeddings = self.rssm.state_action(state_action_input)  # [candidates, sa_dim]

                # Update hidden states for all candidates
                rnn_inputs = torch.cat([state_action_embeddings, current_hiddens], dim=-1).unsqueeze(1)  # [candidates, 1, features]
                _, current_hiddens = self.rssm.rnn(rnn_inputs)
                current_hiddens = current_hiddens.squeeze(1)  # [candidates, hidden_size]

                # Sample from prior for all candidates (deterministic for planning)
                current_states, _, _ = self.rssm.sample_prior(current_hiddens, deterministic=True)

                # Predict rewards for all candidates
                predicted_rewards = self.rssm.reward(
                    torch.cat([current_states, current_hiddens], dim=-1)
                ).squeeze(-1)  # [candidates]

                total_returns += predicted_rewards

        return total_returns

    def _refit_belief(self, top_k_sequences):
        """
        Refit belief to top K action sequences

        Args:
            top_k_sequences: [top_k, horizon, action_dim]

        Returns:
            new_mean: Updated mean [horizon, action_dim]
            new_std: Updated std [horizon, action_dim]
        """
        # Compute empirical mean and std
        new_mean = top_k_sequences.mean(dim=0)
        new_std = top_k_sequences.std(dim=0)

        # Ensure minimum exploration (avoid collapse to deterministic)
        new_std = torch.clamp(new_std, min=0.1)

        return new_mean, new_std


class PlaNetController:
    """Complete PlaNet controller with CEM planning"""

    def __init__(self, rssm, action_dim, horizon=12, action_repeat=2):
        """
        Args:
            rssm: Trained RSSM model
            action_dim: Action space dimensionality
            horizon: Planning horizon
            action_repeat: Number of times to repeat each planned action (R in paper)
        """
        self.rssm = rssm
        # Get device from rssm model
        device = next(rssm.parameters()).device
        self.planner = CEMPlanner(rssm, action_dim, horizon, device=device)
        self.action_repeat = action_repeat

        # State belief tracking
        self.current_state_belief = None
        self.current_hidden = None

        # Action repetition tracking
        self.current_action = None
        self.action_repeat_count = 0

    def reset(self, initial_obs):
        """Reset controller with initial observation"""
        # Encode initial observation
        with torch.no_grad():
            # initial_obs shape: [3, 64, 64], need [1, 3, 64, 64] for encoder
            encoded_obs = self.rssm.encode(initial_obs.unsqueeze(0))

            # Initialize state belief and hidden state
            batch_size = 1
            latent_size = self.rssm.prior_mu.in_features
            hidden_size = self.rssm.rnn.hidden_size

            self.current_state_belief = torch.zeros(batch_size, latent_size)
            self.current_hidden = torch.zeros(batch_size, hidden_size)

            # Update with first observation using posterior
            # encoded_obs shape: [1, 1024], squeeze to [1024] then unsqueeze to [1, 1024] for consistency
            self.current_state_belief, _, _ = self.rssm.sample_posterior(
                self.current_hidden, encoded_obs, deterministic=True
            )

        self.current_action = None
        self.action_repeat_count = 0

    def act(self, obs):
        """
        Get action for current observation

        Args:
            obs: Current observation tensor

        Returns:
            action: Action to take in environment
        """
        # If we're still repeating the previous action
        if self.current_action is not None and self.action_repeat_count < self.action_repeat:
            self.action_repeat_count += 1
            return self.current_action

        # Time to plan a new action
        with torch.no_grad():
            # Encode current observation
            # obs shape: [3, 64, 64], need [1, 3, 64, 64] for encoder
            encoded_obs = self.rssm.encode(obs.unsqueeze(0))

            # Update state belief with current observation (posterior)
            # encoded_obs shape: [1, 1024]
            self.current_state_belief, _, _ = self.rssm.sample_posterior(
                self.current_hidden, encoded_obs, deterministic=True
            )

            # Plan optimal action
            action = self.planner.plan(self.current_state_belief, self.current_hidden)

            self.current_action = action
            self.action_repeat_count = 1

            return action

    def update_state(self, action):
        """Update internal state belief after taking action"""
        with torch.no_grad():
            action_tensor = torch.tensor(action, dtype=torch.float32).unsqueeze(0)

            # Update hidden state using model dynamics
            state_action_embedding = self.rssm.state_action(
                torch.cat([self.current_state_belief, action_tensor], dim=-1)
            )

            rnn_input = torch.cat([state_action_embedding, self.current_hidden], dim=-1).unsqueeze(1)
            _, self.current_hidden = self.rssm.rnn(rnn_input)
            self.current_hidden = self.current_hidden.squeeze(0)

            # Update state belief using prior (no observation yet)
            self.current_state_belief, _, _ = self.rssm.sample_prior(
                self.current_hidden, deterministic=True
            )