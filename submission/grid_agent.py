import torch
import numpy as np


class GridAgent(object):
    def __init__(self, model):

        self.device = torch.device("cuda" if torch.cuda.
                                   is_available() else "cpu")
        self.model = model.to(self.device)

    def predict(self, obs):
        obs = torch.FloatTensor(obs.reshape(1, -1)).to(self.device)
        # print(obs.shape)
        action = self.model.policy(obs)[0]
        # print("ok")
        action_numpy = action.cpu().detach().numpy().flatten()
        return action_numpy

class HybridGridAgent(object):
    def __init__(self, model):

        self.device = torch.device("cuda" if torch.cuda.
                                   is_available() else "cpu")
        self.model = model.to(self.device)

    def predict(self, obs):
        obs = torch.FloatTensor(obs.reshape(1, -1)).to(self.device)
        # print(obs.shape)
        action, _, pi_d = self.model.policy(obs)
        action_d = np.argmax(pi_d, dim=1, keepdim=True)
        action = np.concatenate([action_d, action], 1)
        action_numpy = action.cpu().detach().numpy().flatten()
        return action_numpy
