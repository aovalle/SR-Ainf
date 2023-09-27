# pylint: disable=not-callable
# pylint: disable=no-member

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from pmbrl.utils.misc import one_hot


def swish(x):
    return x * torch.sigmoid(x)


class EnsembleDenseLayer(nn.Module):
    def __init__(self, in_size, out_size, ensemble_size, act_fn="swish"):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.ensemble_size = ensemble_size
        self.act_fn_name = act_fn
        self.act_fn = self._get_act_fn(self.act_fn_name)
        self.reset_parameters()

    def forward(self, x):
        op = torch.baddbmm(self.biases, x, self.weights)
        op = self.act_fn(op)
        return op

    def reset_parameters(self):
        weights = torch.zeros(self.ensemble_size, self.in_size, self.out_size).float()
        biases = torch.zeros(self.ensemble_size, 1, self.out_size).float()

        for weight in weights:
            self._init_weight(weight, self.act_fn_name)

        self.weights = nn.Parameter(weights)
        self.biases = nn.Parameter(biases)

    def _init_weight(self, weight, act_fn_name):
        if act_fn_name == "swish":
            nn.init.xavier_uniform_(weight)
        elif act_fn_name == "linear":
            nn.init.xavier_normal_(weight)

    def _get_act_fn(self, act_fn_name):
        if act_fn_name == "swish":
            return swish
        elif act_fn_name == "linear":
            return lambda x: x


class EnsembleModel(nn.Module):
    def __init__(
        self,
        in_size,
        out_size,
        hidden_size,
        ensemble_size,
        action_size,
        normalizer,
        act_fn="swish",
        device="cpu",
    ):
        super().__init__()

        self.fc_1 = EnsembleDenseLayer(
            in_size, hidden_size, ensemble_size, act_fn=act_fn
        )
        self.fc_2 = EnsembleDenseLayer(
            hidden_size, hidden_size, ensemble_size, act_fn=act_fn
        )
        self.fc_3 = EnsembleDenseLayer(
            hidden_size, hidden_size, ensemble_size, act_fn=act_fn
        )
        self.fc_4 = EnsembleDenseLayer(
            hidden_size, out_size * 2, ensemble_size, act_fn="linear"
        )

        self.ensemble_size = ensemble_size
        self.normalizer = normalizer
        self.action_size = action_size
        self.device = device
        self.max_logvar = -1
        self.min_logvar = -5
        self.device = device
        self.to(device)

    def forward(self, states, actions, batch=False):
        # Split obs from internal measurements
        states, mus = torch.split(states, [states.size(2) - 1, 1], dim=2)
        norm_states, one_hot_actions = self._pre_process_model_inputs(states, actions, batch)
        # Augment state again
        states = torch.cat((norm_states, mus), 2)
        norm_delta_mean, mu_mu, norm_delta_var, mu_var, uni_mean, uni_var = self._propagate_network(
            states, one_hot_actions
        )
        delta_mean, delta_var = self._post_process_model_outputs(
            norm_delta_mean, norm_delta_var
        )
        return delta_mean, mu_mu, delta_var, mu_var

    def loss(self, states, mus, actions, state_deltas, next_mus, batch=False):
        states, actions = self._pre_process_model_inputs(states, actions, batch)           # Normalized states, one-hot actions
        delta_targets = self._pre_process_model_targets(state_deltas)
        # Augment state
        ''' (ensemble, batch, state dim + mu dim) '''
        states = torch.cat((states, mus), 2)
        # Predict (delta) mean and var values of obs and mean and var value of internal state
        delta_mu, mu_mu, delta_var, mu_var, uni_mean, uni_var = self._propagate_network(states, actions)

        # min Negative Log likelihood (= max LL)
        # N/2 log 2πσ² + 1/2σ² Σ_i (x_i - μ)²
        state_err = (delta_mu.detach() - delta_targets.detach()) ** 2
        state_loss = (delta_mu - delta_targets) ** 2 / delta_var + torch.log(delta_var)
        # (original) Mean over states(?), mean over batch, sum ensemble
        state_loss = state_loss.mean(-1).mean(-1).sum()
        state_err = state_err.mean(-1).mean(-1).sum()
        # Mean over states, mean over batch, mean over ensemble
        #loss = loss.mean(-1).mean(-1).mean()

        mu_err = (mu_mu.detach() - next_mus.detach()) ** 2
        mu_loss = (mu_mu - next_mus) ** 2 / mu_var + torch.log(mu_var)
        # (original) Mean over states(?), mean over batch, sum ensemble
        mu_loss = mu_loss.mean(-1).mean(-1).sum()
        mu_err = mu_err.mean(-1).mean(-1).sum()

        # NOTE: unified loss, in case i have to test it
        # targets = torch.cat((delta_targets, next_mus), 2)
        # uni_loss = (uni_mean - targets) ** 2 / uni_var + torch.log(uni_var)
        # # (original) Mean over states(?), mean over batch, sum ensemble
        # uni_loss = uni_loss.mean(-1).mean(-1).sum()
        # print(targets.shape)

        return state_loss, mu_loss, state_err, mu_err

    def sample(self, mean, var):
        return Normal(mean, torch.sqrt(var)).sample()

    def reset_parameters(self):
        self.fc_1.reset_parameters()
        self.fc_2.reset_parameters()
        self.fc_3.reset_parameters()
        self.fc_4.reset_parameters()
        self.to(self.device)

    def _propagate_network(self, states, actions):
        inp = torch.cat((states, actions), dim=2)
        op = self.fc_1(inp)
        op = self.fc_2(op)
        op = self.fc_3(op)
        op = self.fc_4(op)

        # From the output separate means and variances
        delta_mean, delta_logvar = torch.split(op, op.size(2) // 2, dim=2)
        uni_mean = torch.clone(delta_mean)
        # Separate the observation means and the internal states means
        delta_mean, mu_mean = torch.split(delta_mean, [delta_mean.size(2)-1, 1], dim=2)

        delta_logvar = torch.sigmoid(delta_logvar)
        delta_logvar = (
            self.min_logvar + (self.max_logvar - self.min_logvar) * delta_logvar
        )
        delta_var = torch.exp(delta_logvar)
        uni_var = torch.clone(delta_var)

        delta_var, mu_var = torch.split(delta_var, [delta_var.size(2) - 1, 1], dim=2)

        return delta_mean, mu_mean, delta_var, mu_var, uni_mean, uni_var

    def _pre_process_model_inputs(self, states, actions, batch):
        states = states.to(self.device)
        actions = actions.to(self.device)
        states = self.normalizer.normalize_states(states)
        ''' actions (ensemble, batch, action size) '''
        one_hot_actions = one_hot(actions, self.action_size, batch).to(self.device)
        #actions = self.normalizer.normalize_actions(actions)
        return states, one_hot_actions #actions

    def _pre_process_model_targets(self, state_deltas):
        state_deltas = state_deltas.to(self.device)
        state_deltas = self.normalizer.normalize_state_deltas(state_deltas)
        return state_deltas

    def _post_process_model_outputs(self, delta_mean, delta_var):
        delta_mean = self.normalizer.denormalize_state_delta_means(delta_mean)
        delta_var = self.normalizer.denormalize_state_delta_vars(delta_var)
        return delta_mean, delta_var


class RewardModel(nn.Module):
    def __init__(self, in_size, hidden_size, action_size, act_fn="relu", device="cpu"):
        super().__init__()
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.action_size = action_size
        self.device = device
        self.act_fn = getattr(F, act_fn)
        self.reset_parameters()
        self.to(device)

    def forward(self, states, actions):
        inp = torch.cat((states, actions), dim=-1)
        reward = self.act_fn(self.fc_1(inp))
        reward = self.act_fn(self.fc_2(reward))
        reward = self.fc_3(reward).squeeze(dim=1)
        return reward

    # NOTE remove rewards, smirls from fn params
    def loss(self, states, mus, actions, rewards, smirls, combined_rewards, batch=False):
        one_hot_actions = one_hot(actions, self.action_size, batch).to(self.device)
        #r_hat = self(states, actions)
        # Augment state with measurements
        states = torch.cat((states, mus), 2)
        r_hat = self(states, one_hot_actions)
        #return F.mse_loss(r_hat, rewards)
        return F.mse_loss(r_hat, combined_rewards)

    def reset_parameters(self):
        self.fc_1 = nn.Linear(self.in_size, self.hidden_size)
        self.fc_2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc_3 = nn.Linear(self.hidden_size, 1)
        self.to(self.device)
