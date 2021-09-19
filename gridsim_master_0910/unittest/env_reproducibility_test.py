# env utils
import test_utils as utils

# unittest 
import unittest
import numpy as np
from utilize.settings import settings



class TestEnvReproducibility(unittest.TestCase):
    def setUp(self):
        test_repeat = 5 
        np.random.seed(1024)
        self.seeds = np.random.randint(100000, size=test_repeat)
        print("seeds: {}".format(self.seeds))
        self.max_timestep = 288
    
    def _run_multiple_episodes_and_check_result_equal(self, agent, seed=None, create_new_env=False):
        episodes_rewards = []
        episodes_gen_p = []
        episodes_load_p = []
        episodes_rho = []
        sum_rewards = []
        episodes_actions_p = []
        episodes_actions_v = []

        repeat_episodes = 3
        
        env = utils.get_env() 
        for i in range(repeat_episodes):
            
            if create_new_env:
                env = utils.get_env()

            obs = env.reset(seed=seed)

            cur_episode_rewards = []
            cur_episode_gen_p = []
            cur_episode_load_p = []
            cur_episode_rho = []
            cur_sum_reward = 0.0
            cur_actions_p = []
            cur_actions_v = []
            
            cur_episode_gen_p.append(obs.gen_p)
            cur_episode_load_p.append(obs.load_p)
            cur_episode_rho.append(obs.rho)
            
            reward = 0
            done = False
            for _ in range(self.max_timestep):
                action = agent.act(obs, reward, done)
                
                obs, reward, done, info = env.step(action)
                cur_sum_reward += reward

                cur_episode_rewards.append(reward)
                cur_episode_gen_p.append(obs.gen_p)
                cur_episode_load_p.append(obs.load_p)
                cur_episode_rho.append(obs.rho)
                cur_actions_p.append(action['adjust_gen_p'])
                cur_actions_v.append(action['adjust_gen_v'])

                if done:
                    break
            
            episodes_rewards.append(cur_episode_rewards)  
            episodes_gen_p.append(cur_episode_gen_p)  
            episodes_load_p.append(cur_episode_load_p)  
            episodes_rho.append(cur_episode_rho)
            sum_rewards.append(cur_sum_reward)
            episodes_actions_p.append(cur_actions_p)
            episodes_actions_v.append(cur_actions_v)

        
        for i in range(1, repeat_episodes):
            np.testing.assert_equal(np.array(episodes_actions_p[0]), np.array(episodes_actions_p[i]))
            np.testing.assert_equal(np.array(episodes_actions_v[0]), np.array(episodes_actions_v[i]))

            np.testing.assert_equal(np.array(episodes_gen_p[0]), np.array(episodes_gen_p[i]))
            np.testing.assert_equal(np.array(episodes_load_p[0]), np.array(episodes_load_p[i]))

            np.testing.assert_equal(np.array(episodes_rho[0]), np.array(episodes_rho[i]))

            np.testing.assert_equal(np.array(episodes_rewards[0]), np.array(episodes_rewards[i]))
            assert abs(sum_rewards[0] - sum_rewards[i]) < 1e-6
    

    def _reset_multiple_times_and_check_result(self, agent, timestep):
        all_gen_p = []
        all_load_p = []

        repeat_episodes = 3

        env = utils.get_env()

        for i in range(repeat_episodes):
            obs = env.reset(start_sample_idx=timestep)

            all_gen_p.append(obs.gen_p)
            all_load_p.append(obs.load_p)

        for i in range(1, repeat_episodes):
            np.testing.assert_equal(np.array(all_gen_p[0]), np.array(all_gen_p[i]))
            np.testing.assert_equal(np.array(all_load_p[0]), np.array(all_load_p[i]))


    def test_same_seed_reset_multiple_times_get_same_result(self):
        for seed in self.seeds:
            print("seed: {}".format(seed))
            # action_space sample agent
            agent = utils.get_random_agent(seed=seed)
            self._run_multiple_episodes_and_check_result_equal(seed=seed, agent=agent,
                    create_new_env=False)

    def test_same_seed_create_multiple_envs_get_same_result(self):
        for seed in self.seeds:
            print("seed: {}".format(seed))
            # action_space sample agent
            agent = utils.get_random_agent(seed=seed)
            self._run_multiple_episodes_and_check_result_equal(seed=seed, agent=agent,
                    create_new_env=True)

    def test_same_reset_timestep_get_same_data(self):
        for seed in self.seeds:
            print("seed: {}".format(seed))
            np.random.seed(seed)
            timestep = np.random.randint(settings.num_sample)

            agent = utils.get_random_agent(seed=seed)
            self._reset_multiple_times_and_check_result(agent, timestep)



   
if __name__ == '__main__':
    unittest.main()
