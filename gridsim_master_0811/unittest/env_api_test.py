# env utils
import test_utils as utils

# unittest 
import unittest
import numpy as np
import logging
from utilize.settings import settings

class TestEnvAPIs(unittest.TestCase):
    def test_reset_API_support_timestep_argument_4_1(self):
        env = utils.get_env() 
         
        for i in range(10):
            timestep = np.random.randint(settings.num_sample)
            obs = env.reset(start_sample_idx=timestep)

        with self.assertRaises(AssertionError):
            obs = env.reset(start_sample_idx=-1)

        with self.assertRaises(AssertionError):
            obs = env.reset(start_sample_idx=settings.num_sample)



    def test_step_API_will_raise_exception_when_action_dim_is_wrong_4_4(self):
        env = utils.get_env() 
        
        obs = env.reset()
        wrong_action_1 = {}
        with self.assertRaises(AssertionError):
            obs, reward, done, info = env.step(wrong_action_1)



        obs = env.reset()
        wrong_action_2 = {'adjust_gen_p': 0, 'adjust_gen_v': 1}
        with self.assertRaises(AssertionError):
            obs, reward, done, info = env.step(wrong_action_2)

        obs = env.reset()
        wrong_action_3 = {'adjust_gen_p': [0] * 10, 'adjust_gen_v': [0] * 10}
        with self.assertRaises(AssertionError):
            obs, reward, done, info = env.step(wrong_action_3)


    def test_step_API_after_env_is_done_4_6(self):
        env = utils.get_env() 
        agent = utils.get_random_agent()

        obs = env.reset()
        done = False 
        while not done:
            action = agent.act(obs)
            obs, reward, done, info = env.step(action)

        assert done
    
        action = agent.act(obs)
        with self.assertRaises(Exception):
            env.step(action)

if __name__ == '__main__':
    unittest.main()

