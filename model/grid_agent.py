import torch
import numpy as np
import parl

class SAC_GridAgent(parl.Agent):
    def __init__(self, algorithm, device='cpu'):

        super(SAC_GridAgent, self).__init__(algorithm)

        # self.device = torch.device("cuda" if torch.cuda.
        #                            is_available() else "cpu")
        self.device = torch.device(device)
        self.alg.sync_target(decay=0)

    def predict(self, obs):
        obs = torch.FloatTensor(obs.reshape(1, -1)).to(self.device)
        action = self.alg.predict(obs)
        action_numpy = action.cpu().detach().numpy().flatten()
        return action_numpy

    def sample(self, obs):
        obs = torch.FloatTensor(obs.reshape(1, -1)).to(self.device)
        action, *_ = self.alg.sample(obs)
        action_numpy = action.cpu().detach().numpy().flatten()
        return action_numpy

    def learn(self, obs, action, reward, next_obs, terminal):
        terminal = np.expand_dims(terminal, -1)
        reward = np.expand_dims(reward, -1)

        obs = torch.FloatTensor(obs).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        next_obs = torch.FloatTensor(next_obs).to(self.device)
        terminal = torch.FloatTensor(terminal).to(self.device)
        critic_loss, actor_loss = self.alg.learn(obs, action, reward, next_obs,
                                                 terminal)
        return critic_loss, actor_loss