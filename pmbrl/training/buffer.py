# pylint: disable=not-callable
# pylint: disable=no-member

import torch
import numpy as np


class Buffer(object):
    def __init__(
        self,
        state_size,
        action_size,
        ensemble_size,
        normalizer,
        signal_noise=None,
        buffer_size=10 ** 6,
        n_batches=10,
        device="cpu",
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.ensemble_size = ensemble_size
        self.buffer_size = buffer_size
        self.n_batches = n_batches
        self.signal_noise = signal_noise
        self.device = device

        self.states = np.zeros((buffer_size, state_size))
        self.mus = np.zeros((buffer_size, 1))
        self.actions = np.zeros((buffer_size, action_size))
        self.rewards = np.zeros((buffer_size, 1))
        self.smirls = np.zeros((buffer_size, 1))
        self.combined_rewards = np.zeros((buffer_size, 1))
        self.state_deltas = np.zeros((buffer_size, state_size))
        self.mu_deltas = np.zeros((buffer_size, 1))
        self.next_mu = np.zeros((buffer_size, 1))

        self.normalizer = normalizer
        self._total_steps = 0

    # NOTE: should i norm mu, should i use delta mu?
    def add(self, state, mu, action, reward, smirl, combined_reward, next_state, next_mu):
        idx = self._total_steps % self.buffer_size
        state_delta = next_state - state
        mu_delta = next_mu - mu

        self.states[idx] = state
        self.mus[idx] = mu
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.smirls[idx] = smirl
        self.combined_rewards[idx] = combined_reward
        self.state_deltas[idx] = state_delta
        self.mu_deltas[idx] = mu_delta
        self.next_mu[idx] = next_mu
        self._total_steps += 1

        self.normalizer.update(state, action, state_delta)

    def get_train_batches(self, batch_size):
        size = len(self)
        indices = [
            np.random.permutation(range(size)) for _ in range(self.ensemble_size)
        ]
        indices = np.stack(indices).T

        curr_batch = 0
        for i in range(0, size, batch_size):
            j = min(size, i + batch_size)

            if curr_batch >= self.n_batches:
                break

            if (j - i) < batch_size and i != 0:
                return

            batch_size = j - i

            batch_indices = indices[i:j]
            batch_indices = batch_indices.flatten()

            states = self.states[batch_indices]
            mus = self.mus[batch_indices]
            actions = self.actions[batch_indices]
            rewards = self.rewards[batch_indices]
            smirls = self.smirls[batch_indices]
            combined_rewards = self.combined_rewards[batch_indices]
            state_deltas = self.state_deltas[batch_indices]
            mu_deltas = self.mu_deltas[batch_indices]
            next_mus = self.next_mu[batch_indices]

            states = torch.from_numpy(states).float().to(self.device)
            mus = torch.from_numpy(mus).float().to(self.device)
            actions = torch.from_numpy(actions).long().to(self.device)
            rewards = torch.from_numpy(rewards).float().to(self.device)
            combined_rewards = torch.from_numpy(combined_rewards).float().to(self.device)
            smirls = torch.from_numpy(smirls).float().to(self.device)
            state_deltas = torch.from_numpy(state_deltas).float().to(self.device)
            next_mus = torch.from_numpy(next_mus).float().to(self.device)

            if self.signal_noise is not None:
                states = states + self.signal_noise * torch.randn_like(states)

            states = states.reshape(self.ensemble_size, batch_size, self.state_size)
            mus = mus.reshape(self.ensemble_size, batch_size, 1)
            actions = actions.reshape(self.ensemble_size, batch_size, self.action_size)
            rewards = rewards.reshape(self.ensemble_size, batch_size, 1)
            smirls = smirls.reshape(self.ensemble_size, batch_size, 1)
            combined_rewards = combined_rewards.reshape(self.ensemble_size, batch_size, 1)
            state_deltas = state_deltas.reshape(self.ensemble_size, batch_size, self.state_size)
            next_mus = next_mus.reshape(self.ensemble_size, batch_size, 1)

            curr_batch +=1
            yield states, mus, actions, rewards, smirls, combined_rewards, state_deltas, next_mus

    def __len__(self):
        return min(self._total_steps, self.buffer_size)

    @property
    def total_steps(self):
        return self._total_steps
