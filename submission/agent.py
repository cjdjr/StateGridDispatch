import os
# os.environ['PARL_BACKEND'] = 'torch'
# import sys
# sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/../")
import torch
import numpy as np
from submission.grid_model import GridModel
from submission.grid_agent import GridAgent

# from submission.utils import feature_process, action_process
from utils import feature_process, action_process

OBS_DIM = 927
ACT_DIM = 54


class Agent(object):

    def __init__(self, settings, this_directory_path):
        self.settings = settings
        
        model_path = os.path.join(this_directory_path, "saved_model/clip-checkpoint-2050215.tar")

        model = GridModel(OBS_DIM, ACT_DIM)
        
        #torch.save(model.state_dict(), model_path)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))

        self.agent = GridAgent(model)
        
    def act(self, obs, reward, done=False):
        features = self._process_obs(obs)
        # print("finish process obs")
        action = self.agent.predict(features)
        action = 1.5 * np.tanh(action)
        # print("finish predict")
        self.action = action
        ret_action = self._process_action(obs, action)
        # print("finish process act")
        return ret_action
    
    def _process_obs(self, obs):

        return feature_process(self.settings, obs)
    
    def _process_action(self, obs, action):
        
        return action_process(self.settings, obs, action)


if __name__=="__main__":
    model = GridModel(OBS_DIM, ACT_DIM)
    print("ok")