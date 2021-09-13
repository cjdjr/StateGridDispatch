import os
# os.environ['PARL_BACKEND'] = 'torch'
# import sys
# sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/../")
import torch
import numpy as np
from submission.grid_model import GridModel
from submission.grid_agent import GridAgent

def wrap_action(adjust_gen_p):
    act = {
        'adjust_gen_p': adjust_gen_p,
        'adjust_gen_v': np.zeros_like(adjust_gen_p)
    }
    return act

OBS_DIM = 927
ACT_DIM = 54


class Agent(object):

    def __init__(self, settings, this_directory_path):
        self.settings = settings
        
        model_path = os.path.join(this_directory_path, "saved_model/checkpoint-2650237.tar")

        model = GridModel(OBS_DIM, ACT_DIM)
        
        #torch.save(model.state_dict(), model_path)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))

        self.agent = GridAgent(model)
        
    def act(self, obs, reward, done=False):
        features = self._process_obs(obs)
        # print("finish process obs")
        action = self.agent.predict(features)
        action = np.tanh(action)
        # print("finish predict")
        self.action = action
        ret_action = self._process_action(obs, action)
        # print("finish process act")
        return ret_action
    
    def _process_obs(self, obs):

        loads = []
        loads.append(obs.load_p)
        loads.append(obs.load_q)
        loads.append(obs.load_v)
        loads = np.concatenate(loads)

        # prods
        prods = []
        prods.append(obs.gen_p)
        prods.append(obs.gen_q)
        prods.append(obs.gen_v)
        prods = np.concatenate(prods)
        
        # rho
        rho = np.array(obs.rho) - 1.0

        next_load = obs.nextstep_load_p

        # action_space
        action_space_low = obs.action_space['adjust_gen_p'].low.tolist()
        action_space_high = obs.action_space['adjust_gen_p'].high.tolist()
        for id in self.settings.renewable_ids:
            action_space_low[id] = action_space_high[id]
        action_space_low[self.settings.balanced_id] = 0.0
        action_space_high[self.settings.balanced_id] = 0.0
        
        # steps_to_reconnect_line = obs.steps_to_reconnect_line.tolist()
        steps_to_recover_gen = obs.steps_to_recover_gen.tolist()
        # gen_status = obs.gen_status.tolist()
        # 1 stands for can be opened
        gen_status = ((obs.gen_status == 0) & (obs.steps_to_recover_gen == 0)).astype(float).tolist()
        steps_to_close_gen = obs.steps_to_close_gen.tolist()

        gen_features = np.concatenate([gen_status, prods, action_space_low, action_space_high, steps_to_recover_gen])
        gen_features = np.transpose(gen_features.reshape((7,-1))).reshape(7*self.settings.num_gen)
        
        features = np.concatenate([
            gen_features.tolist(),
            loads,
            rho.tolist(), next_load
            # gen_status
        ])

        return features
    
    def _process_action(self, obs, action):
        N = len(action)
        gen_status = ((obs.gen_status == 0) & (obs.steps_to_recover_gen == 0)).astype(float)
        idx = ((action <=0 ) & (gen_status == 1))
        action[np.where(idx==1)] = -1

        gen_p_action_space = obs.action_space['adjust_gen_p']

        low_bound = gen_p_action_space.low
        high_bound = gen_p_action_space.high

        # for i in range(len(low_bound)):
        #     if obs.gen_p[i]>=self.settings['min_gen_p'][i]:
        #         low_bound[i]= max(self.settings['min_gen_p'][i] + 1e-6 - obs.gen_p[i], low_bound[i])

        for id in self.settings.renewable_ids:
            low_bound[id] = high_bound[id]

        mapped_action = low_bound + (action - (-1.0)) * (
            (high_bound - low_bound) / 2.0)
        mapped_action[self.settings.balanced_id] = 0.0
        mapped_action = np.clip(mapped_action, low_bound, high_bound)
        
        return wrap_action(mapped_action)


if __name__=="__main__":
    model = GridModel(OBS_DIM, ACT_DIM)
    print("ok")