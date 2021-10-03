import os
# os.environ['PARL_BACKEND'] = 'torch'
# import sys
# sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/../")
import torch
import numpy as np
from submission.grid_model import GridModel
from submission.grid_agent import GridAgent
from submission.utils import feature_process, action_process
def wrap_action(adjust_gen_p):
    act = {
        'adjust_gen_p': adjust_gen_p,
        'adjust_gen_v': np.zeros_like(adjust_gen_p)
    }
    return act

OBS_DIM = 819 + 54
ACT_DIM = 54
num_ensemble = 3

class Agent(object):

    def __init__(self, settings, this_directory_path):
        self.settings = settings
        
        # model_path = os.path.join(this_directory_path, "saved_model/ensemble_3_1.0_checkpoint_1000042.tar")
        model_path = os.path.join(this_directory_path, "saved_model/ensemble_3_checkpoint_1150126.tar")

        models = []
        for i in range(num_ensemble):
            models.append(GridModel(OBS_DIM, ACT_DIM))
        
        #torch.save(model.state_dict(), model_path)
        state_dict = torch.load(model_path, map_location='cpu')
        assert len(state_dict) == num_ensemble
        for i in range(num_ensemble):
            models[i].load_state_dict(state_dict[i])

        self.agent = GridAgent(models)
        self.obs_statistics = None
        # try:
        #     mean,std = np.load(os.path.join(this_directory_path, "saved_model/obs_statistics.npy"))
        #     self.obs_statistics={
        #         'mean': mean,
        #         'std': std
        #     }
        # except:
        #     pass
        
    def act(self, obs, reward, done=False):
        features = self._process_obs(obs)
        # print("finish process obs")
        action = self.agent.predict(features)
        # print("finish predict")
        self.action = action
        ret_action = self._process_action(obs, action)
        # print("finish process act")
        return ret_action
    
    def _process_obs(self, obs):

        return feature_process(self.settings, obs, self.obs_statistics)
    
    def _process_action(self, obs, action):

        return action_process(self.settings, obs, action)



if __name__=="__main__":
    model = GridModel(OBS_DIM, ACT_DIM)
    print("ok")