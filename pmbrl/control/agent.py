# pylint: disable=not-callable
# pylint: disable=no-member

from copy import deepcopy
from collections import deque

import numpy as np
import torch
import torch.nn as nn


class Agent(object):
    def __init__(self, args, env, planner, logger=None):
        self.env = env
        self.planner = planner
        self.interoception = deque(maxlen=args.mem_size)
        self.smirl_surprise_scale = args.smirl_surprise_scale
        self.smirl_reward_scale = args.smirl_reward_scale
        self.logger = logger

    def get_seed_episodes(self, buffer, n_episodes):
        for _ in range(n_episodes):
            state = self.env.reset()

            # SMiRL
            # D_0 <- {s_0}
            if len(self.interoception) == 0:
                mu = 0.0
                self.interoception.append(np.random.random())
            else:
                mu = np.mean(self.interoception)
                self.interoception.append(1.0)
            #self.interoception.append(1.0)

            done = False
            while not done:
                action = self.env.sample_action()
                next_state, reward, done, _ = self.env.step(action)
                if done:
                    reward = -1
                next_internal_state = 1 - done

                # SMIRL
                # Fit density θ_{t+1} = U(D_t)
                next_mu = np.mean(self.interoception)
                # Reward r_t <- log p_θ_t+1 (s_t+1)
                smirl = self.get_smirl_bern(next_internal_state, next_mu)
                # D_t+1 = D_t U (s_t+1)
                self.interoception.append(next_internal_state)

                combined_reward = self.smirl_reward_scale * reward + self.smirl_surprise_scale * smirl

                buffer.add(state, mu, action, reward, smirl, combined_reward, next_state, next_mu)
                state = deepcopy(next_state)
                mu = deepcopy(next_mu)
                if done:
                    break
        return buffer

    def run_episode(self, buffer=None, action_noise=None, recorder=None):
        total_reward = 0
        total_smirl = 0
        total_comb_reward = 0
        total_steps = 0
        episode_mus = []
        done = False

        with torch.no_grad():
            state = self.env.reset()
            t = 0

            # SMiRL
            # θ_t
            mu = np.mean(self.interoception)
            # D_0 < - {s_0}
            self.interoception.append(1.0)

            # Augment state s0 <- (s0, θ_t, t)
            state = np.concatenate((state, [mu]), axis=0)
            # For stats
            episode_mus.append(mu)

            while not done:

                # Planning ----- Rolling Horizon
                action, pred_return = self.planner.select_action(state)
                print(pred_return)

                if action_noise is not None:
                    action = self._add_action_noise(action, action_noise)
                action = action.cpu().detach().numpy()

                # Act in env
                try:
                    next_state, reward, done, _ = self.env.step(action)
                except:
                    action = action[0]
                    next_state, reward, done, _ = self.env.step(action)
                if done:
                    reward = -1
                next_internal_state = 1 - done

                # SMIRL
                # Fit density θ_{t+1} = U(D_t)
                next_mu = np.mean(self.interoception)
                # Reward r_t <- log p_θ_t+1 (s_t+1)
                smirl = self.get_smirl_bern(next_internal_state, next_mu)
                # D_t+1 = D_t U (s_t+1)
                self.interoception.append(next_internal_state)

                combined_reward = self.smirl_reward_scale*reward + self.smirl_surprise_scale*smirl

                # Separate obs from measurements
                #np.split(state, [state.size-1, state.size]) # Behaves different from torch. less index control and returns empty array
                state, mu = state[:-1], state[-1:]
                # Add to buffer
                buffer.add(state, mu, action, reward, smirl, combined_reward, next_state, next_mu)

                total_reward += reward
                total_smirl += smirl
                total_comb_reward += combined_reward
                total_steps += 1
                t += 1

                if self.logger is not None and total_steps % 25 == 0:
                    self.logger.log(
                        "> Step {} [reward {:.2f}] [smirl {:.2f}] [combined {:.2f}] [epi mean {:.2f}]".format(
                            total_steps, total_reward, total_smirl, total_comb_reward, np.mean(episode_mus))
                    )

                if recorder is not None:
                    recorder.capture_frame()

                state = deepcopy(next_state)
                mu = deepcopy(next_mu)
                # Augment state st <- (st, θ_t, t)
                state = np.concatenate((state, [mu]), axis=0)
                # For stats
                episode_mus.append(mu)
                if done:
                    break

        if recorder is not None:
            recorder.close()
            del recorder

        self.env.close()
        stats = self.planner.return_stats()
        return total_reward, total_smirl, total_comb_reward, episode_mus, self.interoception, total_steps, stats

    def _add_action_noise(self, action, noise):
        if noise is not None:
            action = action + noise * torch.randn_like(action)
        return action

    # log PMF of a bernoulli RV
    def get_smirl_bern_bak(self, obs, mu):
        # Avoid log(0)
        mu = np.clip(mu, 1e-8, 1 - 1e-8)
        # if mu == 0:
        #     mu += 1e-8
        # elif mu == 1:
        #     mu -= 1e-8
        logp = obs*np.log(mu) + (1-obs)*np.log(1-mu)
        return logp

    ''' return -log p'''
    def get_smirl_bern(self, obs, mu):
        # Avoid log(0)
        mu = np.clip(mu, 1e-8, 1 - 1e-8)
        # if mu == 0:
        #     mu += 1e-8
        # elif mu == 1:
        #     mu -= 1e-8
        logp = obs*np.log(mu) + (1-obs)*np.log(1-mu)
        return -logp