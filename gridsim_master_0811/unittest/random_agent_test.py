# -*- coding: UTF-8 -*-
import numpy as np

from Agent.RandomAgent import RandomAgent
from Environment.base_env import Environment
from utilize.settings import settings

def run_task(my_agent):

    for episode in range(max_episode):
        print('------ episode ', episode)
        env = Environment(settings, "EPRIReward")
        print('------ reset ')
        obs = env.reset(seed=episode)

        reward = 0.0
        done = False
        
        # while not done:
        for timestep in range(max_timestep):
            print('------ step ', timestep)
            action = my_agent.act(obs, reward, done)
            # print("adjust_gen_p: ", action['adjust_gen_p'])
            # print("adjust_gen_v: ", action['adjust_gen_v'])
            obs, reward, done, info = env.step(action)
            print('info:', info)
            if done:
                break

if __name__ == "__main__":
    max_timestep = 100  # 最大时间步数
    max_episode = 10  # 回合数

    my_agent = RandomAgent(settings.num_gen)

    run_task(my_agent)
