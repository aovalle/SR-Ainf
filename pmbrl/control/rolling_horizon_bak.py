'''
11/05/2020
RHE created to plan in discrete action spaces for the Expected Future Free Energy
More concretely, this RHE is weighted not by instrumental utility alone but also
by an epistemic component.
In order to compute an epistemic utility it relies on the existence of an ensemble.
Thus this versions of RHE handles ensembles of learned transition models on sequences
of actions (horizon length, number of candidates, action dimensionality).
Different from other RHE versions, after performing a simulated rollout it must return
the sequence of states visited along the trajectory in order to establish comparisons
within the ensemble to compute epistemic metrics.
'''

import numpy as np
import torch
import torch.nn.functional as F
import copy

from pmbrl.control.measures import InformationGain, Disagreement, Variance, Random
from pmbrl.utils.misc import one_hot

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class RollingHorizon():

    def __init__(self, args, transition_model, reward_model, action_space, action_dim):
        self.args = args
        self.transition_model = transition_model
        self.reward_model = reward_model
        self.horizon = args.plan_horizon
        self.optim_iters = args.optimisation_iters
        self.candidates = args.n_candidates
        self.mutation_rate = args.mutation_rate
        self.shift_buffer_on = args.shift_buffer_on

        self.ensemble_size = args.ensemble_size
        self.action_space = action_space        # Repertoire of actions
        self.action_dim = action_dim             # Dimensionality without one-hot encoding

        self.instrumental = args.use_reward
        self.epistemic = args.use_exploration
        self.use_mean = args.use_mean
        self.epistemic_scale = args.expl_scale
        self.instrumental_scale = args.reward_scale

        #self.action_space = action_space
        self.curr_rollout = None

        if args.expl_strategy == 'information':
            self.measure = InformationGain(self.transition_model, scale=self.epistemic_scale)
        elif args.expl_strategy == "variance":
            self.measure = Variance(self.transition_model, scale=self.epistemic_scale)
        elif args.expl_strategy == "random":
            self.measure = Random(self.transition_model, scale=self.epistemic_scale)
        elif args.expl_strategy == "none":
            self.use_exploration = False

        self.trial_rewards = []
        self.trial_bonuses = []
        self.trial_bonuses_mu = []

    def select_action(self, state):
        state = torch.from_numpy(state).float().to(device)

        # Repeat init state for all ensemble for all candidates
        ''' (state dim) -> (ensemble, candidates, state dim) '''
        state = state.unsqueeze(0).unsqueeze(0)
        state = state.repeat(self.ensemble_size, self.candidates, 1)

        # Generate an initial sequence unless shift buffer is on (or there's no current seq)
        if self.curr_rollout is None or not self.shift_buffer_on:
            self.curr_rollout = np.random.randint(self.action_space, size=self.horizon)
        else:
            self.curr_rollout = self.shift_buffer(np.copy(self.curr_rollout))

        highest_return = float("-inf")
        # Number of generations to eval before selecting a sequence
        for _ in range(self.optim_iters):

            # Mutate rollout and generate all action sequences
            ''' (candidates, horizon) '''
            rollouts = self.mutate(self.curr_rollout)
            ''' Fron np to torch (horizon, candidates, action dim) '''
            rollouts = torch.LongTensor(rollouts.transpose()).unsqueeze(-1).to(device)

            # Evaluate rollouts and get best one
            ''' best rollout (horzion, action dim) '''
            best_rollout, best_return = self.evaluate_sequences(state, rollouts)

            if best_return >= highest_return:
                highest_return = best_return
                overall_best_rollout = best_rollout
                self.curr_rollout = best_rollout.cpu().numpy().flatten().astype(int)

        # if highest_return == float("-inf"):
        #     print(best_return, highest_return)
        #     raise

        return overall_best_rollout[0], highest_return


    def evaluate_sequences(self, init_state, action_seqs):
        state_size = init_state.size(2)  # Get state dimension

        # Pass on actions
        # Simulate trajectory and obtain predicted states
        ''' (horizon, ensemble, candidates, dim) '''
        states, deltas_vars, delta_means, mu_vars, mu_means = self.simulate_rollout(init_state, action_seqs)

        returns = torch.zeros(self.candidates).to(device)
        # Epistemic utility
        if self.epistemic:
            # Return epistemic reward of each action-seq candidate
            epistemic_rew = self.measure(delta_means, deltas_vars) * self.epistemic_scale
            # NOTE: for now, let's not do anything other than track it
            epistemic_rew_mu = self.measure(mu_means, mu_vars) * self.epistemic_scale

            # NOTE: check for nans
            epistemic_rew[torch.isnan(epistemic_rew)] = -9.9e+36

            returns += epistemic_rew
            self.trial_bonuses.append(epistemic_rew)
            self.trial_bonuses_mu.append(epistemic_rew_mu)

        # Instrumental utility
        if self.instrumental:
            ''' -> (horizon*ensemble*candidates, dim) '''
            _states = states.view(-1, state_size)
            ''' -> (ensemble*horizon*candidates, dim) '''
            _actions = action_seqs.unsqueeze(0).repeat(self.ensemble_size, 1, 1, 1)
            _actions = _actions.view(-1, self.action_dim)
            _actions = one_hot(_actions, self.action_space, batch=True).to(device)
            # Get predicted rewards
            rewards = self.reward_model(_states, _actions) * self.instrumental_scale
            rewards = rewards.view(self.horizon, self.ensemble_size, self.candidates)
            # Average over ensembles and sum on the trajectory of each candidate
            ''' (candidates) '''
            rewards = rewards.mean(dim=1).sum(dim=0)

            #NOTE: check for nans
            rewards[torch.isnan(rewards)] = -9.9e+36

            returns += rewards
            self.trial_rewards.append(rewards)

        # Obtain best candidate
        best_return, idx_best = returns.max(dim=0)
        best_action_seq = action_seqs[:, idx_best, :]
        return best_action_seq, best_return


    def simulate_rollout(self, init_state, action_seqs):
        T = self.horizon + 1
        # List of tensors to hold values fo each t
        states = [torch.empty(0)] * T
        delta_means = [torch.empty(0)] * T
        mu_means = [torch.empty(0)] * T
        delta_vars = [torch.empty(0)] * T
        mu_vars = [torch.empty(0)] * T

        # Init state sequence
        states[0] = init_state

        # Clone action-seq for all ensemble
        ''' (horizon, candidates, action dim) -> (horizon, ensemble, candidates, action dim)'''
        action_seqs = action_seqs.unsqueeze(0)
        action_seqs = action_seqs.repeat(self.ensemble_size, 1, 1, 1).permute(1, 0, 2, 3)

        # Obtain predicted transition deltas along trajectory
        for t in range(self.horizon):
            delta_means[t + 1], mu_means[t + 1], delta_vars[t + 1], mu_vars[t + 1] = self.transition_model(states[t], action_seqs[t], batch=True)
            # Split obs from internal measurements
            state, mu = torch.split(states[t], [states[t].size(2) - 1, 1], dim=2)

            if self.use_mean:
                states[t + 1] = state + delta_means[t + 1]
            else:
                states[t + 1] = state + self.transition_model.sample(delta_means[t + 1], delta_vars[t + 1])
                mu_means[t + 1] = self.transition_model.sample(mu_means[t + 1], mu_vars[t + 1])

            ''' (ensemble, candidates, dim) '''
            # Recover augmented state
            states[t + 1] = torch.cat((states[t + 1], mu_means[t + 1]), axis=2)

        ''' (horizon, ensemble, candidates, dim) '''
        states = torch.stack(states[1:], dim=0)
        delta_vars = torch.stack(delta_vars[1:], dim=0)
        delta_means = torch.stack(delta_means[1:], dim=0)
        mu_means = torch.stack(mu_means[1:], dim=0)
        mu_vars = torch.stack(mu_vars[1:], dim=0)

        return states, delta_vars, delta_means, mu_vars, mu_means

    def mutate(self, rollout):
        # Clone sequence
        rollouts = np.tile(rollout, (self.candidates-1, 1))
        # Generate indices to mutate
        idx = np.random.rand(*rollouts.shape) < self.mutation_rate
        # Generate new actions and place them accordingly
        rollouts[idx] = np.random.randint(self.action_space, size=len(idx[idx == True]))
        # Add original sequence to mutated ones
        rollouts = np.row_stack((rollout, rollouts))
        return rollouts

    def shift_buffer(self, rollout):
        # append new random action at the end
        sf_rollout = np.append(rollout, np.random.randint(self.action_space))
        # remove first action
        sf_rollout = np.delete(sf_rollout, 0)
        return sf_rollout

    ''' Returns statistics over all candidate solutions generated during an episode '''
    def return_stats(self):
        if self.instrumental:
            reward_stats = self._create_stats(self.trial_rewards)
        else:
            reward_stats = {}
        if self.epistemic:
            info_stats = self._create_stats(self.trial_bonuses)
            info_stats_mu = self._create_stats(self.trial_bonuses_mu)
        else:
            info_stats = {}
            info_stats_mu = {}
        self.trial_rewards = []
        self.trial_bonuses = []
        self.trial_bonuses_mu = []
        return reward_stats, info_stats, info_stats_mu

    def _create_stats(self, arr):
        tensor = torch.stack(arr)
        tensor = tensor.view(-1)
        return {
            "max": tensor.max().item(),
            "min": tensor.min().item(),
            "mean": tensor.mean().item(),
            "std": tensor.std().item(),
        }