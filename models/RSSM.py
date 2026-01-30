import torch
import torch.nn as nn
from models.EncoderDecoder import Encoder, Decoder
import torch.nn.functional as F

class RSSM(nn.Module):
    def __init__(self, action_size : int, sa_dim : int, latent_size : int, encoded_size : int, hidden_size : int, lstm_layers : int):
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

        # lstm is usde for predicting the next hidden state as the "deterministic" component of this setup
        # input should be composed of the previous hidden state, the previous state, and the previous action
        # since the action is going to be one-hot encoded, we can assume that we are going to embed the
        # action and the previous state together using some learned function. we'll call this the sa_dim
        self.rnn = nn.GRU(sa_dim + hidden_size, hidden_size, lstm_layers, dropout = 0.1)

        self.state_action = nn.Sequential(
            nn.Linear(latent_size + action_size, 200),
            nn.ReLU(),
            nn.Linear(200, sa_dim)
        )

        self.prior = nn.Sequential(
            nn.Linear(hidden_size, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU()
        )
        self.prior_mu = nn.Linear(200, latent_size)
        self.prior_logvar = nn.Linear(200, latent_size)

        self.posterior = nn.Sequential(
            nn.Linear(hidden_size + encoded_size, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU()
        )
        self.posterior_mu = nn.Linear(200, latent_size)
        self.posterior_logvar = nn.Linear(200, latent_size)

        self.reward = nn.Sequential(
            nn.Linear(latent_size + hidden_size, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 1)
        )

    def reparameterize(self, mu, logvar, deterministic=False):
        if deterministic:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def sample_prior(self, hidden_state, deterministic=False):
        prior_features = self.prior(hidden_state)
        prior_mu = self.prior_mu(prior_features)
        prior_logvar = self.prior_logvar(prior_features)
        prior_sample = self.reparameterize(prior_mu, prior_logvar, deterministic)
        return prior_sample, prior_mu, prior_logvar

    def sample_posterior(self, hidden_state, encoded_obs, deterministic=False):
        posterior_input = torch.cat([hidden_state, encoded_obs], dim=-1)
        posterior_features = self.posterior(posterior_input)
        posterior_mu = self.posterior_mu(posterior_features)
        posterior_logvar = self.posterior_logvar(posterior_features)
        posterior_sample = self.reparameterize(posterior_mu, posterior_logvar, deterministic)
        return posterior_sample, posterior_mu, posterior_logvar

    def encode(self, obs):
        return self.encoder(obs)

    def decode(self, hidden_state, posterior_state_est):
        return self.decoder(torch.cat([hidden_state, posterior_state_est], dim=-1))
    
    def pass_through(self, prev_stochastic_state, prev_hidden, encoded_obs, actions):
        # We need to have the previous stochastic state because we need to use it to generate the next hidden state
        # The same is true for the prev_hidden state.
        # we'll assume here that both the encoded states and the actions are tensors
        # actions is composed of some number of batches, some number of timesteps, and obviously 1-hot encoded
        B, T, _ = actions.size()

        posterior_states_list = [prev_stochastic_state]
        prior_states_list = [prev_stochastic_state]
        hiddens_list = [prev_hidden]

        prior_mus = []
        prior_logvars = []
        posterior_mus = []
        posterior_logvars = []
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

            state_action_t = self.state_action(torch.cat([posterior_state_t, action_t], dim=-1))

            # Update hidden state using GRU
            rnn_input = torch.cat([state_action_t, hidden_t], dim=-1).unsqueeze(1)  # Add seq dimension: [B, 1, features]
            _, hidden_t = self.rnn(rnn_input)  # hidden_t: [num_layers, B, hidden_size]
            hidden_t = hidden_t.squeeze(0)  # Remove layer dimension: [B, hidden_size]
            hiddens_list.append(hidden_t)

            # Given this new hidden state, let's find the prior and posterior
            # We'll start with predicting the prior
            prior_state_t, prior_mu_t, prior_logvar_t = self.sample_prior(hidden_t)

            # We'll also predict the posterior given this new hidden state
            posterior_state_t, posterior_mu_t, posterior_logvar_t = self.sample_posterior(hidden_t, encoded_obs_t)

            # Predict the reward for a given hidden and posterior state
            reward = self.reward(torch.cat([posterior_state_t, hidden_t], dim=-1))

            # Store results
            prior_states_list.append(prior_state_t)
            prior_mus.append(prior_mu_t)
            prior_logvars.append(prior_logvar_t)

            posterior_states_list.append(posterior_state_t)
            posterior_mus.append(posterior_mu_t)
            posterior_logvars.append(posterior_logvar_t)
            rewards.append(reward)

        prior_states = torch.stack(prior_states_list[1:], dim=1)
        posterior_states = torch.stack(posterior_states_list[1:], dim=1)
        hiddens = torch.stack(hiddens_list[1:], dim=1)
        prior_mus = torch.stack(prior_mus, dim=1)
        prior_logvars = torch.stack(prior_logvars, dim=1)
        posterior_mus = torch.stack(posterior_mus, dim=1)
        posterior_logvars = torch.stack(posterior_logvars, dim=1)
        rewards = torch.stack(rewards, dim = 1)
        return prior_states, posterior_states, hiddens, prior_mus, prior_logvars, posterior_mus, posterior_logvars, rewards