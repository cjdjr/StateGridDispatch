import torch
# import paddle
# import paddle.nn as nn
# import paddle.nn.functional as F
import torch.nn as nn
import torch.nn.functional as F
import parl

class GridModel(parl.Model):
    def __init__(self, obs_dim, action_dim, flags):
        super(GridModel, self).__init__()
        self.gen_input_dim = flags.gen_input_dim
        self.embedding_dim = flags.embedding_dim
        self.gen_num = flags.gen_num

        self.gen_projection_layer = nn.Linear(self.gen_input_dim, self.embedding_dim)
        self.others_projection_layer = nn.Linear(obs_dim - self.gen_num * self.gen_input_dim, 256)
        self.l1 = nn.Linear(self.gen_num * self.embedding_dim + 256, 256)

        self.actor_head = ActorModel(256, action_dim, flags)
        self.critic_head = CriticModel(obs_dim, action_dim, flags)

    def _get_core(self, obs):
        core = []

        gen = obs[:,:self.gen_num * self.gen_input_dim].view(-1, self.gen_num, self.gen_input_dim)
        gen_embedding = self.gen_projection_layer(gen).view(-1, self.gen_num * self.embedding_dim)
        core.append(gen_embedding)

        others = obs[:,self.gen_num * self.gen_input_dim:]
        others_embedding = self.others_projection_layer(others)
        core.append(others_embedding)

        core = torch.cat(core, 1)
        core = self.l1(core)

        return core

    def policy(self, obs):
        core = self._get_core(obs)
        return self.actor_head(core)

    def value(self, obs, action):
        # core = self._get_core(obs)
        # return self.critic_head(core, action)
        return self.critic_head(obs, action)

    def get_actor_params(self):
        ignored_params = list(map(id, self.critic_head.parameters()))
        params = filter(lambda p: id(p) not in ignored_params, self.parameters())
        return params
        # return self.actor_model.get_weights()

    def get_critic_params(self):
        # ignored_params = list(map(id, self.actor_head.parameters()))
        # params = filter(lambda p: id(p) not in ignored_params, self.parameters())
        # return params
        return self.critic_head.parameters()
        # return self.critic_model.get_weights()

class ActorModel(parl.Model):
    def __init__(self, obs_dim, action_dim, flags):
        super(ActorModel, self).__init__()
        self.flags = flags
        # self.l1 = nn.Linear(obs_dim, 512)
        # self.l2 = nn.Linear(512, 256)
        self.mean_linear = nn.Linear(obs_dim, action_dim)
        self.std_linear = nn.Linear(obs_dim, action_dim)

    def forward(self, x):
        # x = obs
        # x = F.relu(self.l1(x))
        # x = F.relu(self.l2(x))

        act_mean = self.mean_linear(x)
        act_std = self.std_linear(x)
        act_log_std = torch.clamp(act_std, min=self.flags.LOG_SIG_MIN, max=self.flags.LOG_SIG_MAX)
        return act_mean, act_log_std


class CriticModel(parl.Model):
    def __init__(self, obs_dim, action_dim, flags):
        super(CriticModel, self).__init__()

        # Q1 network
        self.l1 = nn.Linear(obs_dim + action_dim, 512)
        self.l2 = nn.Linear(512, 256)
        self.l3 = nn.Linear(256, 1)
        # self.l1 = nn.Linear(obs_dim + action_dim, 256)
        # self.l2 = nn.Linear(256, 1)

        # Q2 network
        self.l4 = nn.Linear(obs_dim + action_dim, 512)
        self.l5 = nn.Linear(512, 256)
        self.l6 = nn.Linear(256, 1)
        # self.l3 = nn.Linear(obs_dim + action_dim, 256)
        # self.l4 = nn.Linear(256, 1)

    def forward(self, obs, action):

        x1 = torch.cat([obs, action], 1)
        x2 = torch.cat([obs, action], 1)

        # Q1
        q1 = F.relu(self.l1(x1))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        # Q2
        q2 = F.relu(self.l4(x2))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2