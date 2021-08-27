import unittest
import numpy as np
import pandas as pd
from Environment.base_env import Environment
from Agent.RandomAgent import RandomAgent
from utilize.settings import settings


class DataReadingTest(unittest.TestCase):
    def test_grid_readdata(self):
        seed = 1024
        np.random.seed(seed)
        start_sample_ids = [np.random.randint(settings.num_sample) for _ in range(10)]

        load_p_data = pd.read_csv(settings.load_p_filepath).values.tolist()
        
        env = Environment(settings)
        agent = RandomAgent(num_gen=settings.num_gen, seed=seed)
        for idx in start_sample_ids:
            obs = env.reset(start_sample_idx=idx, seed=seed)
            np.testing.assert_almost_equal(obs.load_p, load_p_data[idx], decimal=2)

            action = agent.act(obs)

            obs, reward, done, info = env.step(action)
            if done:
                continue

            np.testing.assert_almost_equal(obs.load_p, load_p_data[idx + 1], decimal=2)


    def test_read_renewable_gen_p_max(self):
        seed = 1024
        np.random.seed(seed)
        start_sample_ids = [np.random.randint(settings.num_sample) for _ in range(10)]

        renewable_gen_p_max_data = pd.read_csv(settings.max_renewable_gen_p_filepath).values.tolist()
        
        env = Environment(settings)
        agent = RandomAgent(num_gen=settings.num_gen, seed=seed)
        for idx in start_sample_ids:
            obs = env.reset(start_sample_idx=idx, seed=seed)
            np.testing.assert_almost_equal(obs.curstep_renewable_gen_p_max, renewable_gen_p_max_data[idx], decimal=2)
            if idx < settings.num_sample - 1:
                np.testing.assert_almost_equal(obs.nextstep_renewable_gen_p_max, renewable_gen_p_max_data[idx + 1], decimal=2)

            action = agent.act(obs)

            obs, reward, done, info = env.step(action)
            if done:
                continue

            np.testing.assert_almost_equal(obs.curstep_renewable_gen_p_max, renewable_gen_p_max_data[idx + 1], decimal=2)
            if idx + 1 < settings.num_sample - 1:
                np.testing.assert_almost_equal(obs.nextstep_renewable_gen_p_max, renewable_gen_p_max_data[idx + 2], decimal=2)


if __name__ == '__main__':
    unittest.main()
