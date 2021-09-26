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

from pickle import DEFAULT_PROTOCOL
import parl
import torch
from torch.distributions import Normal
import torch.nn.functional as F
from copy import deepcopy
import numpy as np
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
        action = torch.tanh(act_mean)
        return action

    def sample(self, obs):
        act_mean, act_log_std = self.model.policy(obs)
        normal = Normal(act_mean, act_log_std.exp())
        # for reparameterization trick  (mean + std*N(0,1))
        x_t = normal.rsample()
        action = torch.tanh(x_t)

        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log((1 - action.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdims=True)
        return action, log_prob

    def learn(self, obs, action, reward, next_obs, terminal):
        critic_loss = self._critic_learn(obs, action, reward, next_obs,
                                         terminal)
        actor_loss = self._actor_learn(obs)

        self.sync_target()
        return critic_loss, actor_loss

    def _critic_learn(self, obs, action, reward, next_obs, terminal):
        with torch.no_grad():
            next_action, next_log_pro = self.sample(next_obs)
            q1_next, q2_next = self.target_model.critic_model(
                next_obs, next_action)
            target_Q = torch.min(q1_next, q2_next) - self.alpha * next_log_pro
            target_Q = reward + self.gamma * (1. - terminal) * target_Q
        cur_q1, cur_q2 = self.model.critic_model(obs, action)

        critic_loss = F.mse_loss(cur_q1, target_Q) + F.mse_loss(
            cur_q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        return critic_loss

    def _actor_learn(self, obs):
        act, log_pi = self.sample(obs)
        q1_pi, q2_pi = self.model.critic_model(obs, act)
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

class Ensemble_SAC(parl.Algorithm):
    def __init__(self,
                 models,
                 gamma=None,
                 tau=None,
                 autotune=None,
                 alpha=None,
                 actor_lr=None,
                 critic_lr=None,
                 temperature=None,
                 ber_mean=None,
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
        self.autotune = autotune
        self.alpha = alpha
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.temperature = temperature
        self.ber_mean = ber_mean
        device = torch.device(device)
        self.num_ensemble = len(models)
        self.model, self.target_model, self.actor_optimizer, self.critic_optimizer = [], [], [], []
        for i in range(self.num_ensemble):
            model = models[i].to(device)
            target_model = deepcopy(model)
            actor_optimizer = torch.optim.Adam(
                model.get_actor_params(), lr=actor_lr)
            critic_optimizer = torch.optim.Adam(
                model.get_critic_params(), lr=critic_lr)
            self.model.append(model)
            self.target_model.append(target_model)
            self.actor_optimizer.append(actor_optimizer)
            self.critic_optimizer.append(critic_optimizer)

        if self.autotune:
            self.target_entropy = self.alpha
            self.alpha = [1.] * self.num_ensemble
            self.alpha_optimizer, self.log_alpha = [], []
            for i in range(self.num_ensemble):
                log_alpha = torch.zeros(1, requires_grad=True, device=device)
                alpha_optimizer = torch.optim.Adam([log_alpha], lr=actor_lr)
                self.alpha_optimizer.append(alpha_optimizer)
                self.log_alpha.append(log_alpha)
        else:
            self.alpha = [self.alpha] * self.num_ensemble

    def predict(self, obs):
        action = None
        for i in range(self.num_ensemble):
            act_mean, _ = self.model[i].policy(obs)
            act_mean = torch.tanh(act_mean)
            if i==0:
                action = act_mean
            else:
                action += act_mean
        action/=self.num_ensemble
        return action
    
    def sample(self, obs):
        actions , log_probs = [], []
        for i in range(self.num_ensemble):
            act_mean, act_log_std = self.model[i].policy(obs)
            normal = Normal(act_mean, act_log_std.exp())
            # for reparameterization trick  (mean + std*N(0,1))
            x_t = normal.rsample()
            action = torch.tanh(x_t)

            log_prob = normal.log_prob(x_t)
            # Enforcing Action Bound
            log_prob -= torch.log((1 - action.pow(2)) + 1e-6)
            log_prob = log_prob.sum(1, keepdims=True)

            actions.append(action)
            log_probs.append(log_prob)

        return actions, log_probs

    def ucb_sample(self, obs):
        # assume B=1
        actions, ucb_scores = [], []
        for i in range(self.num_ensemble):
            act_mean, act_log_std = self.model[i].policy(obs)
            normal = Normal(act_mean, act_log_std.exp())
            # for reparameterization trick  (mean + std*N(0,1))
            x_t = normal.rsample()
            action = torch.tanh(x_t)
            ucb_score = self._get_ucb_std(obs, action)
            actions.append(action[:,None,:])
            ucb_scores.append(ucb_score)
        # (B, E, D)
        actions = torch.cat(actions, dim=1)
        B, E, D = actions.shape
        # (B, E)
        ucb_scores = torch.cat(ucb_scores, dim=1)
        # (B, 1, D)
        id = torch.argmax(ucb_scores, dim=1)[:,None,None].expand(B,1,D)
        action = torch.gather(actions, 1, id).squeeze(dim=1)

        mask = torch.bernoulli(torch.ones(B, E, device=obs.device).float() * self.ber_mean)

        assert B==1 , "B must be 1"
        if mask.sum() == 0:
            mask[:, np.random.randint(0,E)] = 1.
        return action, mask


    def learn(self, obs, action, reward, next_obs, terminal, masks):
        if self.autotune:
            self._alpha_learn(obs, masks)

        critic_loss = self._critic_learn(obs, action, reward, next_obs,
                                         terminal, masks)
        actor_loss = self._actor_learn(obs, masks)

        self.sync_target()
        return critic_loss, actor_loss

    def _get_ucb_std(self, obs, act):
        Q = []
        mean_Q = None
        for i in range(self.num_ensemble):
            with torch.no_grad():
                q1, q2 = self.model[i].value(obs, act)
                Q.append(q1)
                Q.append(q2)
                if i==0:
                    mean_Q = (q1+q2)/2
                else:
                    mean_Q += (q1+q2)/2

        mean_Q /= self.num_ensemble
        std_Q = None
        for i in range(2*self.num_ensemble):
            if i == 0:
                std_Q = (Q[i].detach() - mean_Q)**2
            else:
                std_Q += (Q[i].detach() - mean_Q)**2
        std_Q /= 2*self.num_ensemble

        ucb_score = mean_Q + torch.sqrt(std_Q).detach()
            
        return ucb_score

    def _corrective_feedback(self, obs, mode=0):
        '''
        mode=0 for actor
        mode=1 for critic
        '''
        mean_Q, std_Q = None, None
        Q = []
        with torch.no_grad():
            actions, _ = self.sample(obs)
            for i in range(self.num_ensemble):
                if mode==0:
                    q1, q2 = self.model[i].value(obs, actions[i])
                else:
                    q1, q2 = self.target_model[i].value(obs, actions[i])
                Q.append(q1)
                Q.append(q2)
                if i==0:
                    mean_Q = (q1+q2)/2
                else:
                    mean_Q += (q1+q2)/2

        mean_Q/=self.num_ensemble
        for i in range(2*self.num_ensemble):
            if i == 0:
                std_Q = (Q[i].detach() - mean_Q)**2
            else:
                std_Q += (Q[i].detach() - mean_Q)**2
        std_Q /= 2*self.num_ensemble

        return torch.sqrt(std_Q).detach()

    def _alpha_learn(self, obs, masks):
        with torch.no_grad():
            actions, log_probs = self.sample(obs)
        for i in range(self.num_ensemble):
            mask = masks[:,i]
            alpha_loss = -(self.log_alpha[i] * (log_probs[i] + self.target_entropy).detach()) * mask
            alpha_loss = alpha_loss.sum() / (mask.sum() + 1)
            self.alpha_optimizer[i].zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer[i].step()
            self.alpha[i] = self.log_alpha[i].detach().exp().item()
        
    def _critic_learn(self, obs, action, reward, next_obs, terminal, masks):
        critic_loss_mean = None
        Q_std = self._corrective_feedback(obs, mode=1)
        weight_target_Q = torch.sigmoid(-Q_std * self.temperature) + 0.5
        with torch.no_grad():
            next_action, next_log_pro = self.sample(next_obs)
        
        for i in range(self.num_ensemble):
            with torch.no_grad():
                q1_next, q2_next = self.target_model[i].critic_model(
                    next_obs, next_action[i])
                target_Q = torch.min(q1_next, q2_next) - self.alpha[i] * next_log_pro[i]
                target_Q = reward + self.gamma * (1. - terminal) * target_Q
            mask = masks[:,i]
            cur_q1, cur_q2 = self.model[i].critic_model(obs, action)
            q1_loss = F.mse_loss(cur_q1, target_Q) * mask * weight_target_Q.detach()
            q2_loss = F.mse_loss(cur_q2, target_Q) * mask * weight_target_Q.detach()
            q1_loss = q1_loss.sum() / (mask.sum()+1)
            q2_loss = q2_loss.sum() / (mask.sum()+1)
            critic_loss = q1_loss + q2_loss
            self.critic_optimizer[i].zero_grad()
            critic_loss.backward()
            self.critic_optimizer[i].step()

            if i==0:
                critic_loss_mean = critic_loss
            else:
                critic_loss_mean += critic_loss

        critic_loss_mean /= self.num_ensemble
        return critic_loss_mean

    def _actor_learn(self, obs, masks):
        actor_loss_mean = None
        act, log_pi = self.sample(obs)
        for i in range(self.num_ensemble):
            q1_pi, q2_pi = self.model[i].critic_model(obs, act[i])
            min_q_pi = torch.min(q1_pi, q2_pi)
            mask = masks[:,i]
            actor_loss = ((self.alpha[i] * log_pi[i]) - min_q_pi) * mask
            actor_loss = actor_loss.sum() / (mask.sum()+1)
            self.actor_optimizer[i].zero_grad()
            actor_loss.backward()
            self.actor_optimizer[i].step()

            if i==0 :
                actor_loss_mean = actor_loss
            else:
                actor_loss_mean += actor_loss

        actor_loss_mean /= self.num_ensemble
        return actor_loss_mean

    def sync_target(self, decay=None):
        if decay is None:
            decay = 1.0 - self.tau
        for i in range(self.num_ensemble):
            for param, target_param in zip(self.model[i].parameters(),
                                        self.target_model[i].parameters()):
                target_param.data.copy_((1 - decay) * param.data +
                                        decay * target_param.data)

    def get_weights(self):
        return [self.model[i].get_weights() for i in range(self.num_ensemble)]

    def set_weights(self, params):
        assert len(params) == self.num_ensemble, "weights' length doesn't match !!! "
        for i in range(self.num_ensemble):
            self.model[i].set_weights(params[i])
    
    def get_state_dict(self):
        return [self.model[i].state_dict() for i in range(self.num_ensemble)]