import numpy as np

# grid env
from Environment.base_env import Environment
from utilize.settings import settings

### define get_env API
def get_env():
    env = Environment(settings)
    return env


def get_random_agent(seed=None):
    from Agent.RandomAgent import RandomAgent
    agent = RandomAgent(settings.num_gen, seed=seed)
    return agent


def change_line_thermal_limit(env, line_id, new_line_thermal_limit):
    print("changing line_thermal_limit, line_id: {}, origin: {}, new: {}".format(
        line_id, env.settings.line_thermal_limit[line_id], new_line_thermal_limit))
    env.settings.line_thermal_limit[line_id] = new_line_thermal_limit
    return env

