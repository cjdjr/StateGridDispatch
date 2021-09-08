# -*- coding: UTF-8 -*-
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/gridsim_master_0811")

ERROR_INFO_LIMIT = 100 # limit 100 characters

class FileNotExistException(Exception):
    def __init__(self, submit_file):
        self.submit_file = submit_file

    def __str__(self):
        return "file {} does not exist.".format(self.submit_file)

class FileFormatWrongException(Exception):
    def __init__(self, submit_file):
        self.submit_file = submit_file

    def __str__(self):
        return "The submission file `{}` is not a zip file.".format(self.submit_file)

class BadZipFileExceptioin(Exception):
    def __str__(self):
        return "The submission zip file cannot be unzipped."

class SubmissionFolderNotExistException(Exception):
    def __str__(self):
        return "`submission` folder cannot be found in your submitted zip file."


class AgentFileNotExistException(Exception):
    def __str__(self):
        return "`agent.py` cannot be found in your submission"

class AgentClassCannotImportException(Exception):
    def __init__(self, error_info):
        self.error_info = error_info

    def __str__(self):
        message = "[`Agent` class in agent.py cannot be imported] {}".format(self.error_info)
        return message[:ERROR_INFO_LIMIT]

class AgentInitException(Exception):
    def __init__(self, error_info):
        self.error_info = error_info

    def __str__(self):
        message = "[Agent init failed] {}".format(self.error_info)
        return message[:ERROR_INFO_LIMIT]

class AgentActException(Exception):
    def __init__(self, error_info):
        self.error_info = error_info

    def __str__(self):
        message = "[Agent act failed] {}".format(self.error_info)
        return message[:ERROR_INFO_LIMIT]

class EnvStepException(Exception):
    def __init__(self, error_info):
        self.error_info = error_info

    def __str__(self):
        message = "[Env step failed] {}".format(self.error_info)
        return message[:ERROR_INFO_LIMIT]

class AgentActTimeout(Exception):
    def __str__(self):
        message = "[Agent act timeout] executing the act function of your agent is out of time limit."
        return message[:ERROR_INFO_LIMIT]

class EvaluationRunTimeout(Exception):
    def __str__(self):
        message = "[Evaluation run timeout] the evaluation of your submission is out of time limit."
        return message[:ERROR_INFO_LIMIT]


#### timeout utils
import signal

class TimeoutException(Exception):
    def __str__(self):
        return "timeout exception."

def handler(signum, frame):
    raise TimeoutException()

class TimeoutContext(object):
    def __init__(self, timeout_s):
        """ Only supported in UNIX

        Args:
            timeout_s(int): seconds of timeout limit
        """
        assert isinstance(timeout_s, int)
        signal.signal(signal.SIGALRM, handler)
        self.timeout_s = timeout_s

    def __enter__(self):
        signal.alarm(self.timeout_s)

    def __exit__(self, type, value, tb):
        # Cancel the timer if the function returned before timeout
        signal.alarm(0)
####
'''
seed : 1234

3300228.pth
steps :    [288.0, 32.0, 288.0, 288.0, 288.0, 288.0, 288.0, 37.0, 288.0, 288.0, 41.0, 288.0, 288.0, 288.0, 288.0, 288.0, 288.0, 288.0, 288.0, 37.0]
scores :    [496.2975680669143, 31.597569051327095, 500.9423893872262, 481.21696899178215, 487.62707878303814, 485.0123982215625, 501.2344270787347, 55.72123536336498, 454.024699994048, 473.5713977843327, 58.14754650788086, 463.80197472167123, 494.537135905989, 488.2880884887793, 493.23378902980096, 492.3303103068049, 493.1923243072265, 485.032625260861, 488.26140603710223, 40.22354333103439]
infos :     [{}, {'fail_info': 'grid is not converged'}, {}, {}, {}, {}, {}, {'fail_info': 'balance gen out of bound'}, {}, {}, {'fail_info': 'balance gen out of bound'}, {}, {}, {}, {}, {}, {}, {}, {}, {'fail_info': 'balance gen out of bound'}]
[Succ]
Score = 398.2147

5750029.pth
steps :    [288.0, 40.0, 288.0, 288.0, 288.0, 288.0, 288.0, 36.0, 288.0, 288.0, 28.0, 288.0, 288.0, 288.0, 288.0, 40.0, 28.0, 288.0, 288.0, 6.0]
scores :    [494.78476895891333, 45.32768837395827, 487.44123329114996, 490.1231273888009, 489.9898952254427, 482.0189564601643, 487.19539577939344, 56.3238784230821, 438.25882988027575, 470.0930020418596, 42.78810862180067, 414.54947183528895, 485.65431839293785, 485.10459610458605, 496.37151229304357, 62.436846900454206, 31.45974262441625, 483.46820376028234, 481.18648738062245, 6.757791025718808]
infos :     [{}, {'fail_info': 'balance gen out of bound'}, {}, {}, {}, {}, {}, {'fail_info': 'balance gen out of bound'}, {}, {}, {'fail_info': 'balance gen out of bound'}, {}, {}, {}, {}, {'fail_info': 'balance gen out of bound'}, {'fail_info': 'balance gen out of bound'}, {}, {}, {'fail_info': 'balance gen out of bound'}]
[Succ]
Score = 346.5667
'''

import os
import copy
import numpy as np


from gridsim_master_0811.Environment.base_env import Environment
from gridsim_master_0811.utilize.settings import settings

steps = []
infos = []
def run_one_episode(env, seed, start_idx, episode_max_steps, agent, act_timeout):
    print("start_idx: ", start_idx)
    obs = env.reset(seed=seed, start_sample_idx=start_idx)

    reward = 0.0
    done = False

    sum_reward = 0.0
    sum_steps = 0.0
    act_timeout_context = TimeoutContext(act_timeout)
    last_obs = None
    last_action = None
    for step in range(episode_max_steps):
        try:
            with act_timeout_context:
                action = agent.act(obs, reward, done)
        except Exception as e:
            if isinstance(e, TimeoutException):
                raise AgentActTimeout()
            raise AgentActException(str(e))
        if step>=39:
            action = last_action
            low_bound = obs.action_space['adjust_gen_p'].low
            high_bound = obs.action_space['adjust_gen_p'].high

            for id in settings.renewable_ids:
                low_bound[id] = high_bound[id]

            action['adjust_gen_p'] = np.clip(action['adjust_gen_p'], low_bound, high_bound)
            pass
        try:
            obs, reward, done, info = env.step(action)
            last_obs = obs
            last_action = action
        except Exception as e:
            if isinstance(e, TimeoutException):
                raise TimeoutException()
            raise EnvStepException(str(e))

        sum_reward += reward['score'] if not done else 0
        sum_steps += 1
        if done:
            break
    print("step: ",sum_steps)
    print("info: ",info)
    steps.append(sum_steps)
    infos.append(info)
    return sum_reward


import tempfile
from zipfile import ZipFile, BadZipFile

def eval(submit_file=None):

    SEED = 0
    ACT_TIMEOUT = 1 # seconds, the time limit of each step
    RUN_TIMEOUT=1800 # seconds, the time limit of the whole evaluation

    from gridsim_master_0811.utilize.settings import settings
    
    try:
        from submission.agent import Agent
    except Exception as e:
        raise AgentClassCannotImportException(str(e))
    
    run_timeout_context = TimeoutContext(RUN_TIMEOUT)
    
    try:
        with run_timeout_context:
            try:
                agent = Agent(copy.deepcopy(settings), "submission/")
            except Exception as e:
                if isinstance(e, TimeoutException):
                    raise TimeoutException()
                raise AgentInitException(str(e))


            env = Environment(settings,'AllReward')

            episode_max_steps = 288
            scores = []
            np.random.seed(1234)
            idx = np.random.randint(settings.num_sample, size=20)
            # 1 -> step = 40
            for start_idx in idx:
                # start_idx = 4654
                score = run_one_episode(env, SEED, start_idx, episode_max_steps, agent, ACT_TIMEOUT)
                scores.append(score)

    
    except Exception as e:
        if isinstance(e, TimeoutException):
            raise EvaluationRunTimeout()
        else:
            raise e

    mean_score = np.mean(scores)
    print("steps :   ",steps)
    print("scores :   ",scores)
    print("infos :    ",infos)
    return {'score': mean_score}

if __name__ == "__main__":
    try:
        score = eval()['score']
        print('[Succ]')
        print('Score = %.4f' % score)
    except Exception as e:
        print('[Fail]')
        print(type(e))
        print(e)
        print('-'*50 + '\n')
