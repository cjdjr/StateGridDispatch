#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import parl
import torch
from torch.distributions import Normal, Categorical
import torch.nn.functional as F
from copy import deepcopy
from utils import get_hybrid_mask
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SAC(parl.Algorithm):
    def __init__(self,
                 model,
                 gamma=None,
                 tau=None,
                 alpha=None,
                 actor_lr=None,
                 critic_lr=None,
                 device='cpu'):
        """ SAC algorithm
            Args:
                model(parl.Model): forward network of actor and critic.
                gamma(float): discounted factor for reward computation
                tau (float): decay coefficient when updating the weights of self.target_model with self.model
                alpha (float): Temperature parameter determines the relative importance of the entropy against the reward
                actor_lr (float): learning rate of the actor model
                critic_lr (float): learning rate of the critic model
        """
        assert isinstance(gamma, float)
        assert isinstance(tau, float)
        assert isinstance(alpha, float)
        assert isinstance(actor_lr, float)
        assert isinstance(critic_lr, float)

        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        device = torch.device(device)
        self.model = model.to(device)
        self.target_model = deepcopy(self.model)
        self.actor_optimizer = torch.optim.Adam(
            self.model.get_actor_params(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(
            self.model.get_critic_params(), lr=critic_lr)

    def predict(self, obs):
        act_mean, _ = self.model.policy(obs)
        action = 1.5 * torch.tanh(act_mean)
        return action

    def sample(self, obs):
        act_mean, act_log_std = self.model.policy(obs)
        normal = Normal(act_mean, act_log_std.exp())
        # for reparameterization trick  (mean + std*N(0,1))
        x_t = normal.rsample()
        action = torch.tanh(x_t)

        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log((1 - action.pow(2)) * 1.5 + 1e-6)
        log_prob = log_prob.sum(1, keepdims=True)
        return action * 1.5, log_prob

    def learn(self, obs, action, reward, next_obs, terminal):
        critic_loss = self._critic_learn(obs, action, reward, next_obs,
                                         terminal)
        actor_loss = self._actor_learn(obs)

        self.sync_target()
        return critic_loss, actor_loss

    def _critic_learn(self, obs, action, reward, next_obs, terminal):
        with torch.no_grad():
            next_action, next_log_pro = self.sample(next_obs)
            q1_next, q2_next = self.target_model.value(
                next_obs, next_action)
            target_Q = torch.min(q1_next, q2_next) - self.alpha * next_log_pro
            target_Q = reward + self.gamma * (1. - terminal) * target_Q
        cur_q1, cur_q2 = self.model.value(obs, action)

        critic_loss = F.mse_loss(cur_q1, target_Q) + F.mse_loss(
            cur_q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        return critic_loss

    def _actor_learn(self, obs):
        act, log_pi = self.sample(obs)
        q1_pi, q2_pi = self.model.value(obs, act)
        min_q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = ((self.alpha * log_pi) - min_q_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        return actor_loss

    def sync_target(self, decay=None):
        if decay is None:
            decay = 1.0 - self.tau
        for param, target_param in zip(self.model.parameters(),
                                       self.target_model.parameters()):
            target_param.data.copy_((1 - decay) * param.data +
                                    decay * target_param.data)


class Hybrid_SAC(parl.Algorithm):
    def __init__(self,
                 model,
                 gamma=None,
                 tau=None,
                 alpha=None,
                 alpha_d=None,
                 actor_lr=None,
                 critic_lr=None,
                 device='cpu'):
        """ SAC algorithm
            Args:
                model(parl.Model): forward network of actor and critic.
                gamma(float): discounted factor for reward computation
                tau (float): decay coefficient when updating the weights of self.target_model with self.model
                alpha (float): Temperature parameter determines the relative importance of the entropy against the reward
                actor_lr (float): learning rate of the actor model
                critic_lr (float): learning rate of the critic model
        """
        assert isinstance(gamma, float)
        assert isinstance(tau, float)
        assert isinstance(alpha, float)
        assert isinstance(actor_lr, float)
        assert isinstance(critic_lr, float)

        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.alpha_d = alpha_d
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        device = torch.device(device)
        self.model = model.to(device)
        self.target_model = deepcopy(self.model)
        self.actor_optimizer = torch.optim.Adam(
            self.model.get_actor_params(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(
            self.model.get_critic_params(), lr=critic_lr)

    def predict(self, obs):
        mask = get_hybrid_mask(obs)
        B = obs.shape[0]
        mask = torch.cat([mask, torch.ones(B, 1, device = obs.device).float()], dim=1)
        act_mean, _, pi_d = self.model.policy(obs, mask)
        action = torch.tanh(act_mean)
        action_d = torch.argmax(pi_d, dim=1, keepdim=True)
        return torch.cat([action_d, action], dim=1)

    def sample(self, obs):
        mask = get_hybrid_mask(obs)

        act_mean, act_log_std, pi_d = self.model.policy(obs, mask)

        normal = Normal(act_mean, act_log_std.exp())
        # for reparameterization trick  (mean + std*N(0,1))
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log((1 - action.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdims=True)

        dist = Categorical(probs=pi_d)
        action_d = dist.sample()
        prob_d = dist.probs
        log_prob_d = torch.log(prob_d + 1e-6)

        return torch.cat([action_d[:,None], action], dim=1), log_prob_d, log_prob, prob_d

    def learn(self, obs, action, reward, next_obs, terminal):
        critic_loss = self._critic_learn(obs, action, reward, next_obs,
                                         terminal)
        actor_loss = self._actor_learn(obs)

        self.sync_target()
        return critic_loss, actor_loss

    def _critic_learn(self, obs, action, reward, next_obs, terminal):
        with torch.no_grad():
            next_action, next_log_prob_d, next_log_prob_c, next_prob_d = self.sample(next_obs)
            next_action_d, next_action_c = next_action[:,0], next_action[:,1:]
            next_mask = get_hybrid_mask(next_obs)
            q1_next, q2_next = self.target_model.value(
                next_obs, next_action_c, next_mask)
            target_Q = next_prob_d * (torch.min(q1_next, q2_next) - self.alpha * next_prob_d * next_log_prob_c - self.alpha_d * next_log_prob_d)
            target_Q = reward + self.gamma * (1. - terminal) * target_Q
        action_d, action_c = action[:,0], action[:,1:]
        mask = get_hybrid_mask(obs)
        cur_q1, cur_q2 = self.model.value(obs, action_c, mask)

        critic_loss = F.mse_loss(cur_q1, target_Q) + F.mse_loss(
            cur_q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        return critic_loss

    def _actor_learn(self, obs):
        action, log_prob_d, log_prob_c, prob_d = self.sample(obs)
        action_d, action_c = action[:,0], action[:,1:]
        mask = get_hybrid_mask(obs)
        q1_pi, q2_pi = self.model.value(obs, action_c, mask)
        min_q_pi = torch.min(q1_pi, q2_pi)
        actor_loss_d = (prob_d * (self.alpha_d * log_prob_d - min_q_pi)).sum(1).mean()
        actor_loss_c = (prob_d * (self.alpha * prob_d * log_prob_c - min_q_pi)).sum(1).mean()
        actor_loss = actor_loss_d + actor_loss_c

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        return actor_loss

    def sync_target(self, decay=None):
        if decay is None:
            decay = 1.0 - self.tau
        for param, target_param in zip(self.model.parameters(),
                                       self.target_model.parameters()):
            target_param.data.copy_((1 - decay) * param.data +
                                    decay * target_param.data)
