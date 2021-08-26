import torch
import torch.nn as nn
import torch.nn.functional as F

class GridModel(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(GridModel, self).__init__()
        self.l1 = nn.Linear(obs_dim, 512)
        self.l2 = nn.Linear(512, 256)
        self.mean_linear = nn.Linear(256, action_dim)


    def forward(self, obs):
        x = F.relu(self.l1(obs))
        x = F.relu(self.l2(x))

        act_mean = self.mean_linear(x)
        action = torch.tanh(act_mean)
        return action
