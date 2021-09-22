from submission.utils import get_hybrid_mask
import torch
import numpy as np

GEN_NUM = 54

class GridAgent(object):
    def __init__(self, model):

        self.device = torch.device("cuda" if torch.cuda.
                                   is_available() else "cpu")
        self.model = model.to(self.device)

    def predict(self, obs):
        obs = torch.FloatTensor(obs.reshape(1, -1)).to(self.device)
        # print(obs.shape)
        action = torch.tanh(self.model.policy(obs)[0])
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
        mask = get_hybrid_mask(obs)

        action, _, pi_d = self.model.policy(obs, mask)
        action = torch.tanh(action)
        action_d = torch.argmax(pi_d, dim=1, keepdim=True)
        action = torch.cat([action_d, action], dim=1)
        action_numpy = action.cpu().detach().numpy().flatten()
        return action_numpy
