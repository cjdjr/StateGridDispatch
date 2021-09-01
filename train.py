import os
os.environ['PARL_BACKEND'] = 'torch'
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/gridsim_master_0811")

import logging
import hydra
from omegaconf import OmegaConf, DictConfig
import wandb
import numpy as np
import torch
import threading
import time
import parl
from parl.utils import ReplayMemory
# from parl.algorithms import SAC
from parl.utils.window_stat import WindowStat

from gridsim_master_0811.Environment.base_env import Environment
from gridsim_master_0811.utilize.settings import settings

from model.grid_agent import SAC_GridAgent
from model.grid_model import GridModel
from model.env_wrapper import wrap_env
from model.algo import SAC
# from gridsim_master_0811.env_wrapper import get_env
def get_common_flags(flags):
    flags = OmegaConf.to_container(flags)
    flags["SAVE_DIR"] = os.getcwd()
    return OmegaConf.create(flags)

def get_learner_flags(flags):
    lrn_flags = OmegaConf.to_container(flags)
    lrn_flags["checkpoint"] = os.path.join(flags["SAVE_DIR"], "checkpoint.tar")
    return OmegaConf.create(lrn_flags)

def get_env():
    env = Environment(settings, "EPRIReward")
    env = wrap_env(env, settings)
    return env

@parl.remote_class
class Actor(object):
    def __init__(self, flags):
        # print("ok")
        self.env =  get_env()
        obs_dim = flags.OBS_DIM
        action_dim = flags.ACT_DIM
        self.action_dim = action_dim
        self.flags = flags
        # Initialize model, algorithm, agent, replay_memory
        
        model = GridModel(obs_dim, action_dim, flags)
        # print(model.get_actor_params())
        # print(model.get_critic_params())
        algorithm = SAC(
            model,
            gamma=flags.GAMMA,
            tau=flags.TAU,
            alpha=flags.ALPHA,
            actor_lr=flags.ACTOR_LR,
            critic_lr=flags.CRITIC_LR,
            device='cpu')

        self.agent = SAC_GridAgent(algorithm)
        self.do_nothing_action = np.zeros(flags.ACT_DIM) # The adjustments of power generators are zeros.


    
    def sample(self, weights, random_action):
        # sample one episode

        self.agent.set_weights(weights)
        obs = self.env.reset()
        done = False
        episode_training_reward, episode_env_reward, episode_steps = 0, 0, 0
        sample_data = []
        
        # MDP RL
        while not done:
            # Select action randomly or according to policy
            if random_action:
                action = np.random.uniform(-1, 1, size=self.action_dim)
            else:
                action = self.agent.sample(obs)

            # Perform action
            next_obs, reward, done, info = self.env.step(action)
            terminal = done and not info['timeout']
            terminal = float(terminal)

            sample_data.append((obs, action, reward, next_obs, terminal))

            obs = next_obs
            episode_training_reward += reward
            episode_env_reward += info['origin_reward']
            episode_steps += 1

        # # Semi-MDP RL
        
        # # jump to the first overflowed obs 
        # while not done:
        #     if not self.env.has_emergency:
        #         # Expert rule: use do-nothing action when the grid doesn't have overflow lines.
        #         next_obs, reward, done, info = self.env.step(self.do_nothing_action)
        #         obs = next_obs
                
        #         episode_env_reward += info['origin_reward']
        #         episode_training_reward += reward
        #         episode_steps += 1
        #     else:
        #         break
                
        # while not done:       
        #     # Select action randomly or according to policy
        #     if random_action:
        #         action = np.random.uniform(-1, 1, size=self.action_dim)
        #     else:
        #         action = self.agent.sample(obs)

        #     # Perform action
        #     next_obs, reward, done, info = self.env.step(action)
        #     cumulative_discounted_reward = reward
            
        #     episode_env_reward += info['origin_reward']
        #     episode_training_reward += reward
        #     episode_steps += 1
            
        #     # jump to the next overflowed obs 
        #     while not done:
        #         step = 0
        #         if not self.env.has_emergency:
        #             # Expert rule: use do-nothing action when the grid doesn't have overflow lines.
        #             step += 1
        #             next_obs, reward, done, info = self.env.step(self.do_nothing_action)
        #             cumulative_discounted_reward += (self.flags.GAMMA ** step) * reward
                    
        #             episode_env_reward += info['origin_reward']
        #             episode_training_reward += reward
        #             episode_steps += 1
        #         else:
        #             break
                    
        #     terminal = done and not info['timeout']
        #     terminal = float(terminal)
        #     sample_data.append((obs, action, cumulative_discounted_reward, next_obs, terminal))     

        #     obs = next_obs

        return sample_data, episode_env_reward, episode_training_reward, episode_steps

class Learner(object):
    def __init__(self, flags):
        self.model_lock = threading.Lock()
        self.rpm_lock = threading.Lock()
        self.log_lock = threading.Lock()
        
        self.flags = flags
        obs_dim = self.flags.OBS_DIM
        action_dim = self.flags.ACT_DIM

        # Initialize model, algorithm, agent, replay_memory
        model = GridModel(obs_dim, action_dim, flags)
        algorithm = SAC(
            model,
            gamma=flags.GAMMA,
            tau=flags.TAU,
            alpha=flags.ALPHA,
            actor_lr=flags.ACTOR_LR,
            critic_lr=flags.CRITIC_LR,
            device="cuda" if torch.cuda.is_available() else "cpu")
        self.agent = SAC_GridAgent(algorithm)
        self.rpm = ReplayMemory(
            max_size=flags.MEMORY_SIZE, obs_dim=obs_dim, act_dim=action_dim)

        self.total_steps = 0          # 环境交互的所有步数
        self.total_MDP_steps = 0      # 强化模型决策的所有步数
        self.mean_episode_training_reward = WindowStat(100)
        self.mean_episode_env_reward = WindowStat(100)
        self.mean_episode_steps = WindowStat(100)

        self.save_cnt = 0
        self.log_cnt = 0

       
        for _ in range(flags.ACTOR_NUM):
            th = threading.Thread(target=self.run_sampling)
            th.setDaemon(True)
            th.start()

    def checkpoint(self, checkpoint_path=None):
        if self.flags.checkpoint:
            if checkpoint_path is None:
                checkpoint_path = self.flags.checkpoint
            logging.info("Saving checkpoint to %s", checkpoint_path)
            torch.save(self.agent.alg.model.state_dict(),checkpoint_path)

    def run_sampling(self):
        actor = Actor(self.flags)
        while True:
            start = time.time()
            weights = None
            with self.model_lock:
                weights = self.agent.get_weights()

            random_action = False
            if self.rpm.size() < self.flags.WARMUP_STEPS:
                random_action = True

            sample_data, episode_env_reward, episode_training_reward, episode_steps = actor.sample(weights, random_action)

            # Store data in replay memory
            with self.rpm_lock:
                for data in sample_data:
                    self.rpm.append(*data)

            sample_time = time.time() - start
            start = time.time()

            # Train agent after collecting sufficient data
            if self.rpm.size() >= self.flags.WARMUP_STEPS:
                for _ in range(len(sample_data)):
                    with self.rpm_lock:
                        batch_obs, batch_action, batch_reward, batch_next_obs, batch_terminal = self.rpm.sample_batch(
                            self.flags.BATCH_SIZE)
                    with self.model_lock:
                        critic_loss, actor_loss = self.agent.learn(batch_obs, batch_action, batch_reward, batch_next_obs, batch_terminal)
                        if self.flags.wandb:
                            loss = {}
                            loss['critic_loss'] = critic_loss.item()
                            loss['actor_loss'] = actor_loss.item()
                            wandb.log(loss, step=self.total_steps)

            learn_time = time.time() - start

            with self.log_lock:
                self.total_steps += episode_steps
                self.total_MDP_steps += len(sample_data)
                self.mean_episode_env_reward.add(episode_env_reward)
                self.mean_episode_training_reward.add(episode_training_reward)
                self.mean_episode_steps.add(episode_steps)

                if self.total_steps // self.flags.LOG_EVERY_STEPS > self.log_cnt:
                    while self.total_steps // self.flags.LOG_EVERY_STEPS > self.log_cnt:
                        self.log_cnt += 1
                    stats = {}
                    stats['step'] = self.total_steps
                    stats['total_MDP_steps'] = self.total_MDP_steps
                    stats['mean_episode_env_reward'] = self.mean_episode_env_reward.mean
                    stats['mean_episode_training_reward'] = self.mean_episode_training_reward.mean
                    stats['mean_episode_steps'] = self.mean_episode_steps.mean

                    # tensorboard.add_scalar('mean_episode_env_reward', self.mean_episode_env_reward.mean, self.total_steps)
                    # tensorboard.add_scalar('mean_episode_steps', self.mean_episode_steps.mean, self.total_steps)
                    # tensorboard.add_scalar('total_MDP_steps', self.total_MDP_steps, self.total_steps)
                    if self.flags.wandb:
                        wandb.log(stats, step=stats["step"])
                    logging.info('Total Steps: {} Total MDP steps: {} mean_episode_env_rewards: {} mean_episode_steps: {}'.format(
                        self.total_steps, self.total_MDP_steps, self.mean_episode_env_reward.mean, self.mean_episode_steps.mean))

                if self.total_steps // self.flags.SAVE_EVERY_STEPS > self.save_cnt:
                    while self.total_steps // self.flags.SAVE_EVERY_STEPS > self.save_cnt:
                        self.save_cnt += 1
                    with self.model_lock:
                        self.checkpoint()
                        # logging.info(os.getcwd())
                        # self.agent.save(os.path.join(self.flags.SAVE_DIR, "model-{}".format(self.total_steps)))
                        # self.agent.save(os.path.join(self.flags.SAVE_DIR, "model-{}".format(self.total_steps)))

                if self.total_steps > self.flags.MAX_STEPS:
                    break

@hydra.main(config_name="config")
def main(flags: DictConfig):
    if os.path.exists("config.yaml"):
        # this ignores the local config.yaml and replaces it completely with saved one
        logging.info("loading existing configuration, we're continuing a previous run")
        new_flags = OmegaConf.load("config.yaml")
        cli_conf = OmegaConf.from_cli()
        # however, you can override parameters from the cli still
        # this is useful e.g. if you did total_steps=N before and want to increase it
        flags = OmegaConf.merge(new_flags, cli_conf)

    # logging.info("wmr")

    OmegaConf.save(flags, "config.yaml")
    flags = get_common_flags(flags)
    flags = get_learner_flags(flags)
    if flags.wandb:
        wandb.init(
            name=flags.run_name or flags.default_run_name,
            project=flags.project,
            config=vars(flags),
            group=flags.group,
            entity=flags.entity,
        )
    # logging.info(flags.SAVE_DIR)

    # env = Environment(settings, "EPRIReward")
    # env = wrap_env(env, settings)
    parl.connect(
        "localhost:8010",
    )
    time.sleep(10) # wait for connecting

    learner = Learner(flags)
    logging.info("Starting learning....")
    while True:
        time.sleep(1)

if __name__=="__main__":

    os.system("xparl stop")
    os.system("xparl start --port 8010 --monitor_port 8034 --cpu_num 32")
    main()
    pass