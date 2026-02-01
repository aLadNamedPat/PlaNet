import torch
import torch.nn as nn
from models.EncoderDecoder import Encoder, Decoder
import torch.nn.functional as F

class RSSM(nn.Module):
    def __init__(self, action_size : int, latent_size : int, encoded_size : int, hidden_size : int, min_std_dev: float = 0.1, device='cpu'):
        super(RSSM, self).__init__()
        # latent size is going to be the dimension of the latent state (processed from the posterior and prior learned functions)
        # embed size is going to be the "encoded" observation dimensionality 

        # deep help from https://medium.com/@lukasbierling/recurrent-state-space-models-pytorch-implementation-ba5d7e063d11
        # for understanding how the encoder does not directly act as a latent space generator

        # encoder will "encode" image observation into smaller dimensional state
        self.encoder = Encoder()

        # decoder will "decode" from the posterior state estimate and the hidden state to image observation
        # this is largely used for reconstruction loss
        self.decoder = Decoder()

        self.min_std_dev = min_std_dev

        # lstm is usde for predicting the next hidden state as the "deterministic" component of this setup
        # input should be composed of the previous hidden state, the previous state, and the previous action
        # since the action is going to be one-hot encoded, we can assume that we are going to embed the
        # action and the previous state together using some learned function. we'll call this the sa_dim
        # Only use dropout if num_layers > 1
        self.fc_embed_state_action = nn.Linear(latent_size + action_size, hidden_size)
        self.rnn = nn.GRUCell(hidden_size, hidden_size)

        self.prior = nn.Sequential(
            nn.Linear(hidden_size, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU()
        )
        self.prior_mu = nn.Linear(200, latent_size)
        self.prior_std = nn.Linear(200, latent_size)

        self.posterior = nn.Sequential(
            nn.Linear(hidden_size + encoded_size, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU()
        )
        self.posterior_mu = nn.Linear(200, latent_size)
        self.posterior_std = nn.Linear(200, latent_size)

        self.reward = nn.Sequential(
            nn.Linear(latent_size + hidden_size, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 1)
        )

    def sample_prior(self, hidden_state, deterministic=False):
        prior_features = self.prior(hidden_state)
        prior_mu = self.prior_mu(prior_features)
        prior_std_raw = self.prior_std(prior_features)
        
        prior_std = F.softplus(prior_std_raw) + self.min_std_dev
        
        if deterministic:
            return prior_mu, prior_mu, prior_std
        
        prior_sample = prior_mu + prior_std * torch.randn_like(prior_mu)
        return prior_sample, prior_mu, prior_std

    def sample_posterior(self, hidden_state, encoded_obs, deterministic=False):
        posterior_input = torch.cat([hidden_state, encoded_obs], dim=-1)
        posterior_features = self.posterior(posterior_input)
        posterior_mu = self.posterior_mu(posterior_features)
        posterior_std_raw = self.posterior_std(posterior_features)
        
        posterior_std = F.softplus(posterior_std_raw) + self.min_std_dev
        
        if deterministic:
            return posterior_mu, posterior_mu, posterior_std
        
        posterior_sample = posterior_mu + posterior_std * torch.randn_like(posterior_mu)
        return posterior_sample, posterior_mu, posterior_std
    
    def encode(self, obs):
        return self.encoder(obs)

    def decode(self, hidden_state, posterior_state_est):
        # Debug decoder input shapes (only first few times)
        if not hasattr(self, '_decode_debug_count'):
            self._decode_debug_count = 0

        if self._decode_debug_count < 3:
            print(f"        DECODE DEBUG - Hidden: {hidden_state.shape}, Posterior: {posterior_state_est.shape}")
            combined = torch.cat([hidden_state, posterior_state_est], dim=-1)
            print(f"        DECODE DEBUG - Combined input: {combined.shape}")
            self._decode_debug_count += 1

        combined_input = torch.cat([hidden_state, posterior_state_est], dim=-1)
        return self.decoder(combined_input)
    
    def pass_through(self, prev_stochastic_state, prev_hidden, encoded_obs, actions):
        # We need to have the previous stochastic state because we need to use it to generate the next hidden state
        # The same is true for the prev_hidden state.
        # we'll assume here that both the encoded states and the actions are tensors
        # actions is composed of some number of batches, some number of timesteps, and obviously 1-hot encoded
        B, T, _ = actions.size()

        # Debug input shapes (only on first call)
        if not hasattr(self, '_debug_shapes'):
            self._debug_shapes = True
            print(f"    RSSM pass_through - Input shapes:")
            print(f"      prev_stochastic_state: {prev_stochastic_state.shape}")
            print(f"      prev_hidden: {prev_hidden.shape}")
            print(f"      encoded_obs: {encoded_obs.shape}")
            print(f"      actions: {actions.shape}")
            print(f"      B={B}, T={T}")

        posterior_states_list = [prev_stochastic_state]
        prior_states_list = [prev_stochastic_state]
        hiddens_list = [prev_hidden]

        prior_mus = []
        prior_stds = []
        posterior_mus = []
        posterior_stds = []
        rewards = []
        for t in range(T):
            # Again super thankful to https://medium.com/@lukasbierling/recurrent-state-space-models-pytorch-implementation-ba5d7e063d11 for guidance:

            # find the current state, action, etc in each of the batches based on the current timestep
            # encoded state is composed of [B, T, encoded_size]
            encoded_obs_t = encoded_obs[:, t, :]
            action_t = actions[:, t, :]

            
            # we need to select the last timestep because that's realistically the next timestep.
            posterior_state_t = posterior_states_list[-1]
            hidden_t = hiddens_list[-1]

            # Update hidden state using GRU
            embedded = F.relu(self.fc_embed_state_action(
                torch.cat([posterior_state_t, action_t], dim=-1)
            ))
            hidden_t = self.rnn(embedded, hidden_t)  # Clean and simple

            hiddens_list.append(hidden_t)

            # Given this new hidden state, let's find the prior and posterior
            # We'll start with predicting the prior
            prior_state_t, prior_mu_t, prior_std_t = self.sample_prior(hidden_t)

            # We'll also predict the posterior given this new hidden state
            posterior_state_t, posterior_mu_t, posterior_std_t = self.sample_posterior(hidden_t, encoded_obs_t)

            # Predict the reward for a given hidden and posterior state
            reward = self.reward(torch.cat([posterior_state_t, hidden_t], dim=-1))

            # Store results
            prior_states_list.append(prior_state_t)
            prior_mus.append(prior_mu_t)
            prior_stds.append(prior_std_t)

            posterior_states_list.append(posterior_state_t)
            posterior_mus.append(posterior_mu_t)
            posterior_stds.append(posterior_std_t)
            rewards.append(reward)

        prior_states = torch.stack(prior_states_list[1:], dim=1)
        posterior_states = torch.stack(posterior_states_list[1:], dim=1)
        hiddens = torch.stack(hiddens_list[1:], dim=1)
        prior_mus = torch.stack(prior_mus, dim=1)
        prior_stds = torch.stack(prior_stds, dim=1)
        posterior_mus = torch.stack(posterior_mus, dim=1)
        posterior_stds = torch.stack(posterior_stds, dim=1)
        rewards = torch.stack(rewards, dim = 1)
        return prior_states, posterior_states, hiddens, prior_mus, prior_stds, posterior_mus, posterior_stds, rewards