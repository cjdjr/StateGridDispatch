import torch
# import paddle
# import paddle.nn as nn
# import paddle.nn.functional as F
import torch.nn as nn
import torch.nn.functional as F
# import parl

LOG_SIG_MAX = 2.0
LOG_SIG_MIN = -20.0

class GridModel(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(GridModel, self).__init__()
        self.gen_input_dim = 7
        self.embedding_dim = 16
        self.gen_num = 54

        self.gen_projection_layer = nn.Linear(self.gen_input_dim, self.embedding_dim)
        self.gen_embedding = None
        self.gen_status = None
        self.others_projection_layer = nn.Linear(obs_dim - self.gen_num * self.gen_input_dim, 256)
        self.l1 = nn.Linear(self.gen_num * self.embedding_dim + 256, 256)

        self.actor_head = ActorModel(256, action_dim)
        self.critic_head = CriticModel(obs_dim, action_dim)

    def _get_core(self, obs):
        core = []

        # (B, N, input_dim)
        gen = obs[:,:self.gen_num * self.gen_input_dim].view(-1, self.gen_input_dim, self.gen_num).transpose(1,2)
        # (B, N)
        self.gen_status = obs[:,:self.gen_num]
        # (B, N, E)
        self.gen_embedding = F.relu(self.gen_projection_layer(gen))
        core.append(self.gen_embedding.view(-1, self.gen_num * self.embedding_dim))

        others = obs[:,self.gen_num * self.gen_input_dim:]
        others_embedding = F.relu(self.others_projection_layer(others))
        core.append(others_embedding)

        core = torch.cat(core, 1)
        core = F.relu(self.l1(core))

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

class ActorModel(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(ActorModel, self).__init__()
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
        act_log_std = torch.clamp(act_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return act_mean, act_log_std


class CriticModel(nn.Module):
    def __init__(self, obs_dim, action_dim):
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

gen_input_dim = 7
embedding_dim = 16
gen_num = 54
tanh_clipping = 10.

class HybridGridModel(nn.Module):
    def __init__(self, obs_dim, con_action_dim, dis_action_dim):
        super(HybridGridModel, self).__init__()
        self.gen_input_dim = gen_input_dim
        self.embedding_dim = embedding_dim
        self.gen_num = gen_num

        self.gen_projection_layer = nn.Linear(self.gen_input_dim, self.embedding_dim)
        self.gen_embedding = None
        self.gen_status = None
        self.others_projection_layer = nn.Linear(obs_dim - self.gen_num * self.gen_input_dim, 256)
        self.l1 = nn.Linear(self.gen_num * self.embedding_dim + 256, 256)

        self.actor_head = HybridActorModel(256, con_action_dim, dis_action_dim)
        self.critic_head = HybridCriticModel(obs_dim, con_action_dim, dis_action_dim)

    def _get_core(self, obs):
        core = []

        # (B, N, input_dim)
        gen = obs[:,:self.gen_num * self.gen_input_dim].view(-1, self.gen_input_dim, self.gen_num).transpose(1,2)
        # (B, N)
        self.gen_status = obs[:,:self.gen_num]
        # (B, N, E)
        self.gen_embedding = F.relu(self.gen_projection_layer(gen))
        core.append(self.gen_embedding.view(-1, self.gen_num * self.embedding_dim))

        others = obs[:,self.gen_num * self.gen_input_dim:]
        others_embedding = F.relu(self.others_projection_layer(others))
        core.append(others_embedding)

        core = torch.cat(core, 1)
        core = F.relu(self.l1(core))

        return core

    def policy(self, obs, mask):
        core = self._get_core(obs) 
        ninf_mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return self.actor_head(core, ninf_mask)

    def value(self, obs, action, mask):
        # core = self._get_core(obs)
        # return self.critic_head(core, action)
        zero_mask = 1-mask
        return self.critic_head(obs, action, zero_mask)

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

class HybridActorModel(nn.Module):
    def __init__(self, obs_dim, con_action_dim, dis_action_dim):
        super(HybridActorModel, self).__init__()
        # self.l1 = nn.Linear(obs_dim, 512)
        # self.l2 = nn.Linear(512, 256)
        self.mean_linear = nn.Linear(obs_dim, con_action_dim)
        self.std_linear = nn.Linear(obs_dim, con_action_dim)
        self.pi_d_linear = nn.Linear(obs_dim, dis_action_dim)

    def forward(self, x, ninf_mask=None):
        # x = obs
        # x = F.relu(self.l1(x))
        # x = F.relu(self.l2(x))

        act_mean = self.mean_linear(x)
        act_std = self.std_linear(x)
        act_log_std = torch.clamp(act_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)

        score = self.pi_d_linear(x)
        score_clipped = tanh_clipping * torch.tanh(score)
        score_maskd = score_clipped + ninf_mask
        pi_d = F.softmax(score_maskd, dim=-1)
        return act_mean, act_log_std, pi_d

class HybridCriticModel(nn.Module):
    def __init__(self, obs_dim, con_action_dim, dis_action_dim):
        super(HybridCriticModel, self).__init__()

        # Q1 network
        self.l1 = nn.Linear(obs_dim + con_action_dim, 512)
        self.l2 = nn.Linear(512, 256)
        self.l3 = nn.Linear(256, dis_action_dim)
        # self.l1 = nn.Linear(obs_dim + action_dim, 256)
        # self.l2 = nn.Linear(256, 1)

        # Q2 network
        self.l4 = nn.Linear(obs_dim + con_action_dim, 512)
        self.l5 = nn.Linear(512, 256)
        self.l6 = nn.Linear(256, dis_action_dim)
        # self.l3 = nn.Linear(obs_dim + action_dim, 256)
        # self.l4 = nn.Linear(256, 1)

    def forward(self, obs, action, zero_mask=None):

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

        q1 = q1 * (1-zero_mask)
        q2 = q2 * (1-zero_mask)
        return q1, q2