"""
test case: 1.1-1.3
"""

# env utils
import test_utils as utils

# unittest 
import unittest
import numpy as np
import logging
from utilize.settings import settings


class TestGridLogicLineDisconnect(unittest.TestCase):
    def setUp(self):
        test_repeat = 5 
        np.random.seed(1111)
        self.seeds = np.random.randint(100000, size=test_repeat)
        print("seeds: {}".format(self.seeds))
        self.max_timestep = 288
    
    def _check_hard_overflow(self, seed, env, agent):
        np.random.seed(seed)
        chosen_line_id = np.random.randint(settings.num_line)

        ##### get real rho
        obs = env.reset(seed=seed)

        if obs.rho[chosen_line_id] < 1e-5: # line disconnect
            logging.warning("chosen line is disconnected.")
            return False

        if obs.rho[chosen_line_id] > settings.hard_overflow_bound:
            logging.warning("chosen line is already hard overflow, rho: {}".format(obs.rho[chosen_line_id]))
            return False

    
        action = agent.act(obs)
        obs, reward, done, info = env.step(action)
        
        if obs.rho[chosen_line_id] < 1e-5: # line disconnect
            logging.warning("chosen line is disconnected.")
            return False

        if done:
            logging.warning("env is done. ")
            return False


        real_rho = obs.rho[chosen_line_id]
        real_thermal_limit = settings.line_thermal_limit[chosen_line_id]

        ##### change rho to hard_overflow_bound * 1.01 by changing line thermal limit
        expected_rho = settings.hard_overflow_bound * 1.01
        new_thermal_limit = real_rho * real_thermal_limit / expected_rho

        obs = env.reset(seed=seed)

        env = utils.change_line_thermal_limit(env, chosen_line_id, new_thermal_limit)
        action = agent.act(obs)
        obs, reward, done, info = env.step(action)

        assert abs(obs.rho[chosen_line_id] - expected_rho) < 1e-3, (obs.rho[chosen_line_id], expected_rho)

        action = agent.act(obs, reward=0.0)
        obs, reward, done, info = env.step(action)
        
        if done:
            logging.warning("env is done after chosen line is disconnected. info: {}".format(info))
            return False

        assert obs.rho[chosen_line_id] < 1e-5, obs.rho[chosen_line_id] # rho ~= 0
        assert obs.line_status[chosen_line_id] == False
    
        return True

    def test_line_rho_larger_than_hard_overflow_bound_will_disconnect_in_next_step(self):
        valid_cnt = 0
        all_cnt = 0
        for seed in self.seeds:
            env = utils.get_env()
            agent = utils.get_random_agent(seed=seed)

            valid = self._check_hard_overflow(seed=seed, env=env, agent=agent)
            if valid:
                valid_cnt += 1
            all_cnt += 1

        valid_rate = valid_cnt / all_cnt
        print("valid test: {}%".format(valid_rate * 100))
        assert valid_rate > 0.5

    def _check_soft_overflow(self, seed, env, agent):
        np.random.seed(seed)
        chosen_line_id = np.random.randint(settings.num_line)

        ##### get real rho
        obs = env.reset(seed=seed)

        if obs.rho[chosen_line_id] < 1e-5: # line disconnect
            logging.warning("chosen line is disconnected.")
            return False

        if obs.rho[chosen_line_id] > settings.soft_overflow_bound:
            logging.warning("chosen line is already soft overflow, rho: {}".format(obs.rho[chosen_line_id]))
            return False
        
        all_new_thermal_limits = []
        # change rho to soft_overflow_bound * 1.01 by changing line thermal limit
        expected_rho = settings.soft_overflow_bound * 1.01
        for i in range(settings.max_steps_soft_overflow):
            action = agent.act(obs)
            obs, reward, done, info = env.step(action)
            
            if obs.rho[chosen_line_id] < 1e-5: # line disconnect
                logging.warning("chosen line is disconnected.")
                return False

            if done:
                logging.warning("env is done.")
                return False
            
            real_rho = obs.rho[chosen_line_id]
            real_thermal_limit = settings.line_thermal_limit[chosen_line_id]

            
            new_thermal_limit =  real_rho * real_thermal_limit / expected_rho
            all_new_thermal_limits.append(new_thermal_limit)


        #### soft overflow case
        obs = env.reset(seed=seed)

        for i in range(settings.max_steps_soft_overflow):
            new_thermal_limit = all_new_thermal_limits[i]
            env = utils.change_line_thermal_limit(env, chosen_line_id, new_thermal_limit)

            action = agent.act(obs)
            obs, reward, done, info = env.step(action)

            assert abs(obs.rho[chosen_line_id] - expected_rho) < 1e-3, (obs.rho[chosen_line_id], expected_rho)
        
            assert not done 

            assert obs.line_status[chosen_line_id] == True
            assert obs.rho[chosen_line_id] > settings.soft_overflow_bound

        action = agent.act(obs)
        obs, reward, done, info = env.step(action)
        if done:
            logging.warning("env is done after chosen line is disconnected. info: {}".format(info))
            return False

        assert obs.line_status[chosen_line_id] == False
        assert obs.rho[chosen_line_id] < 1e-5 # rho ~= 0

        return True

    def test_line_rho_larger_than_soft_overflow_bound_and_smaller_than_hard_overflow_bound_will_disconnect_after_max_steps_soft_overflow(self):
        valid_cnt = 0
        all_cnt = 0
        for seed in self.seeds:
            env = utils.get_env()
            agent = utils.get_random_agent(seed=seed)

            # start after reset
            valid = self._check_soft_overflow(seed=seed, env=env, agent=agent)
            if valid:
                valid_cnt += 1
            all_cnt += 1

        valid_rate = valid_cnt / all_cnt
        print("valid test: {}%".format(valid_rate * 100))
        assert valid_rate > 0.5
    
    def _check_line_auto_reconnect(self, seed, env, agent):
        np.random.seed(seed)
        chosen_line_id = np.random.randint(settings.num_line)

        ##### get real rho
        obs = env.reset(seed=seed)

        if obs.rho[chosen_line_id] < 1e-5: # line disconnect
            logging.warning("chosen line is disconnected.")
            return False

        if obs.rho[chosen_line_id] > settings.hard_overflow_bound:
            logging.warning("chosen line is already hard overflow, rho: {}".format(obs.rho[chosen_line_id]))
            return False

    
        action = agent.act(obs)
        obs, reward, done, info = env.step(action)
        
        if obs.rho[chosen_line_id] < 1e-5: # line disconnect
            logging.warning("chosen line is disconnected.")
            return False

        if done:
            logging.warning("env is done. ")
            return False


        real_rho = obs.rho[chosen_line_id]
        real_thermal_limit = settings.line_thermal_limit[chosen_line_id]

        ##### change rho to hard_overflow_bound * 1.01 by changing line thermal limit
        expected_rho = settings.hard_overflow_bound * 1.01
        new_thermal_limit = real_rho * real_thermal_limit / expected_rho

        obs = env.reset(seed=seed)

        env = utils.change_line_thermal_limit(env, chosen_line_id, new_thermal_limit)
        action = agent.act(obs)
        obs, reward, done, info = env.step(action)

        assert abs(obs.rho[chosen_line_id] - expected_rho) < 1e-3, (obs.rho[chosen_line_id], expected_rho)

        action = agent.act(obs)
        obs, reward, done, info = env.step(action)
        
        if done:
            logging.warning("env is done after chosen line is disconnected. info: {}".format(info))
            return False

        assert obs.rho[chosen_line_id] < 1e-5, obs.rho[chosen_line_id] # rho ~= 0
        assert obs.line_status[chosen_line_id] == False

        
        for i in range(settings.max_steps_to_reconnect_line - 1):
            action = agent.act(obs)
            obs, reward, done, info = env.step(action)

            assert obs.line_status[chosen_line_id] == False, "disconnect steps: {}".format(i)
            assert obs.rho[chosen_line_id] < 1e-5 # rho ~= 0

            if done:
                logging.warning("done before test case")
                return False
        
        # auto reconnect after 16 steps
        action = agent.act(obs)
        obs, reward, done, info = env.step(action)
        assert obs.line_status[chosen_line_id] == True
        assert obs.rho[chosen_line_id] > 1e-5 # rho > 0

        return True

    def test_disconnected_line_will_be_reconnected_automatically_after_max_steps_to_reconnect_line(self):
        valid_cnt = 0
        all_cnt = 0
        for seed in self.seeds:
            env = utils.get_env()
            agent = utils.get_random_agent(seed=seed)

            valid = self._check_line_auto_reconnect(seed=seed, env=env, agent=agent)
            if valid:
                valid_cnt += 1
            all_cnt += 1


        valid_rate = valid_cnt / all_cnt
        print("valid test: {}%".format(valid_rate * 100))
        assert valid_rate > 0

if __name__ == '__main__':
    unittest.main()
