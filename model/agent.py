import os
import torch
import numpy as np
from model.grid_model import GridModel
from model.grid_agent import SAC_GridAgent


def wrap_action(adjust_gen_p):
    act = {
        'adjust_gen_p': adjust_gen_p,
        'adjust_gen_v': np.zeros_like(adjust_gen_p)
    }
    return act


class Agent(object):

    def __init__(self, settings, this_directory_path, flags=None):
        self.settings = settings
        
        # model_path = os.path.join(this_directory_path, "saved_model/model-0")

        # model = GridModel(OBS_DIM, ACT_DIM)
        
        #torch.save(model.state_dict(), model_path)
        # model.load_state_dict(torch.load(model_path))

        self.agent = SAC_GridAgent(model)
        
    def act(self, obs, reward, done=False):
        features = self._process_obs(obs)
        action = self.agent.predict(features)
        ret_action = self._process_action(obs, action)
        return ret_action
    
    def _process_obs(self, obs):
        # loads
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
        
        features = np.concatenate([loads, prods, rho.tolist()])
        return features
    
    def _process_action(self, obs, action):
        N = len(action)

        gen_p_action_space = obs.action_space['adjust_gen_p']

        low_bound = gen_p_action_space.low
        high_bound = gen_p_action_space.high

        mapped_action = low_bound + (action - (-1.0)) * (
            (high_bound - low_bound) / 2.0)
        mapped_action[self.settings.balanced_id] = 0.0
        mapped_action = np.clip(mapped_action, low_bound, high_bound)
        
        return wrap_action(mapped_action)


    