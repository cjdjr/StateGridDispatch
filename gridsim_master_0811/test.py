# -*- coding: UTF-8 -*-
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


import os
import copy
import numpy as np


from Environment.base_env import Environment

def run_one_episode(env, seed, start_idx, episode_max_steps, agent, act_timeout):
    print("start_idx: ", start_idx)
    obs = env.reset(seed=seed, start_sample_idx=start_idx)

    reward = 0.0
    done = False

    sum_reward = 0.0
    sum_steps = 0.0
    act_timeout_context = TimeoutContext(act_timeout)
    for step in range(episode_max_steps):
        try:
            with act_timeout_context:
                action = agent.act(obs, reward, done)
        except Exception as e:
            if isinstance(e, TimeoutException):
                raise AgentActTimeout()
            raise AgentActException(str(e))

        try:
            obs, reward, done, info = env.step(action)
        except Exception as e:
            if isinstance(e, TimeoutException):
                raise TimeoutException()
            raise EnvStepException(str(e))

        sum_reward += reward
        sum_steps += 1

        if done:
            break

    return sum_reward, sum_steps


import tempfile
from zipfile import ZipFile, BadZipFile

def eval(submit_file):
    ####################################
    # prepare submission agent
    ####################################
    assert isinstance(submit_file, str)

    if not os.path.exists(submit_file):
        raise FileNotExistException(submit_file) 
    if not submit_file.endswith(".zip"):
        raise FileFormatWrongException(submit_file)
    
    tmp_dir = tempfile.TemporaryDirectory()
    tmp_path = tmp_dir.name
    try:
        with ZipFile(submit_file, 'r') as zipObj:
            zipObj.extractall(tmp_path)
    except:
        raise BadZipFileExceptioin()

    submission_path = os.path.join(tmp_path, 'submission') # submission directory

    if not os.path.isdir(submission_path):
        raise SubmissionFolderNotExistException()
    if not os.path.isfile(os.path.join(submission_path, "agent.py")):
        raise AgentFileNotExistException()

    sys.path.append(tmp_path)
    ####################################


    SEED = 0
    ACT_TIMEOUT = 1 # seconds, the time limit of each step
    RUN_TIMEOUT=1800 # seconds, the time limit of the whole evaluation

    from utilize.settings import settings
    
    try:
        from submission.agent import Agent
    except Exception as e:
        raise AgentClassCannotImportException(str(e))
    
    run_timeout_context = TimeoutContext(RUN_TIMEOUT)
    
    try:
        with run_timeout_context:
            try:
                agent = Agent(copy.deepcopy(settings), submission_path)
            except Exception as e:
                if isinstance(e, TimeoutException):
                    raise TimeoutException()
                raise AgentInitException(str(e))


            env = Environment(settings)

            episode_max_steps = 288
            scores = []
            for start_idx in np.random.randint(settings.num_sample, size=20):
                score = run_one_episode(env, SEED, start_idx, episode_max_steps, agent, ACT_TIMEOUT)
                scores.append(score)

    
    except Exception as e:
        if isinstance(e, TimeoutException):
            raise EvaluationRunTimeout()
        else:
            raise e

    mean_score = np.mean(scores)

    return {'score': mean_score}

if __name__ == "__main__":
    import sys
    assert len(sys.argv) == 2
    try:
        score = eval(sys.argv[1])['score']
        print('[Succ]')
        print('Score = %.4f' % score)
    except Exception as e:
        print('[Fail]')
        print(type(e))
        print(e)
        print('-'*50 + '\n')
