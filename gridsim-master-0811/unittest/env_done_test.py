import numpy as np

from Agent.RandomAgent import RandomAgent
from Environment.base_env import Environment
from utilize.settings import settings

class TestEnvDone():
    def test_illegal_action(self):
        print("test illegal actions")
        env = Environment(settings, "EPRIReward")
        obs = env.reset()
        action = my_agent.act(obs, 0, False)
        action['adjust_gen_p'][0] = -10000
        obs, reward, done, info = env.step(action)
        assert done == True
        print(info)

    def test_over_rows(self):
        print("test over rows")
        env = Environment(settings, "EPRIReward")
        obs = env.reset(start_sample_idx = settings.num_sample - 1)
        action = my_agent.act(obs, 0, False)
        obs, reward, done, info = env.step(action)
        assert done == True
        print(info)

if __name__ == "__main__":
    my_agent = RandomAgent(settings.num_gen)
    test = TestEnvDone()
    test.test_illegal_action()
    test.test_over_rows()
