import torch
import numpy as np


class GridAgent(object):
    def __init__(self, models):

        self.device = torch.device("cuda" if torch.cuda.
                                   is_available() else "cpu")
        self.num_ensemble = len(models)
        self.models = []
        for i in range(self.num_ensemble):
            self.models.append(models[i].to(self.device))

    def get_score(self, obs, act):
        score = 0
        for i in range(self.num_ensemble):
            q1, q2 = self.models[i].value(obs,act)
            score += (q1+q2)/2
        return score/self.num_ensemble

    def predict(self, obs):
        obs = torch.FloatTensor(obs.reshape(1, -1)).to(self.device)

        # mean_action = torch.tanh(self.models[0].policy(obs)[0])
        # mean_action = None
        # for i in range(self.num_ensemble):
        #     action = torch.tanh(self.models[i].policy(obs)[0])
        #     if i==0:
        #         mean_action = action
        #     else:
        #         mean_action += action
        # mean_action/=self.num_ensemble
        # action_numpy = mean_action.cpu().detach().numpy().flatten()

        actions = []
        scores = []
        for i in range(self.num_ensemble):
            action = torch.tanh(self.models[i].policy(obs)[0])
            actions.append(action)
            scores.append(self.get_score(obs, action))
        best_act = actions[np.argmax(scores)]
        action_numpy = best_act.cpu().detach().numpy().flatten()

        return action_numpy
