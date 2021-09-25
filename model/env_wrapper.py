import gym
import numpy as np
from parl.utils import logger
from utils import feature_process, action_process

class Wrapper(gym.Env):
    """Wraps the environment to allow a modular transformation.

    This class is the base class for all wrappers. The subclass could override
    some methods to change the behavior of the original environment without touching the
    original code.

    .. note::

        Don't forget to call ``super().__init__(env)`` if the subclass overrides :meth:`__init__`.

    """
    def __init__(self, env):
        self.env = env

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError("attempted to get missing private attribute '{}'".format(name))
        return getattr(self.env, name)

    @property
    def spec(self):
        return self.env.spec

    @classmethod
    def class_name(cls):
        return cls.__name__

    def step(self, action):
        return self.env.step(action)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def render(self, mode='human', **kwargs):
        return self.env.render(mode, **kwargs)

    def close(self):
        return self.env.close()

    def seed(self, seed=None):
        return self.env.seed(seed)

    def compute_reward(self, achieved_goal, desired_goal, info):
        return self.env.compute_reward(achieved_goal, desired_goal, info)

    def __str__(self):
        return '<{}{}>'.format(type(self).__name__, self.env)

    def __repr__(self):
        return str(self)

    @property
    def unwrapped(self):
        return self.env.unwrapped


MAX_TIMESTEP = 288
class MaxTimestepWrapper(Wrapper):
    def __init__(self, env):
        super(MaxTimestepWrapper,self).__init__(env)
        self.timestep = 0

    def step(self, action, **kwargs):
        self.timestep += 1
        obs, reward, done, info = self.env.step(action, **kwargs)
        if self.timestep >= MAX_TIMESTEP:
            done = True
            info["timeout"] = True
        else:
            info["timeout"] = False
        return obs, reward, done, info

    def reset(self, **kwargs):
        self.timestep = 0
        return self.env.reset(**kwargs)


class ObsTransformerWrapper(Wrapper):

    def __init__(self, env, settings, obs_statistics=None):
        super(ObsTransformerWrapper,self).__init__(env)
        self.settings = settings
        self.has_overflow = False # Add an attribute to mark whether the env has overflowed lines.. 
        self.has_overbalance = False
        self.obs_statistics = obs_statistics

    def _get_obs(self, obs):

        return feature_process(self.settings, obs, self.obs_statistics)

    def step(self, action, **kwargs):
        self.raw_obs, reward, done, info = self.env.step(action, **kwargs)
        self.has_overflow = self._has_overflow(self.raw_obs)
        self.has_overbalance = self._has_overbalance(self.raw_obs)
        obs = self._get_obs(self.raw_obs)
        return obs, reward, done, info

    def reset(self, **kwargs):
        self.raw_obs = self.env.reset(**kwargs)
        self.has_overflow = self._has_overflow(self.raw_obs)
        self.has_overbalance = self._has_overbalance(self.raw_obs)
        obs = self._get_obs(self.raw_obs)
        return obs
    
    def _has_overflow(self, obs):
        has_overflow = False
        if obs is not None and not any(np.isnan(obs.rho)):
            has_overflow = any(np.array(obs.rho) > 1.0)
        return has_overflow

    def _has_overbalance(self, obs):
        has_overbalance = False
        balanced_id = self.settings.balanced_id
        max_gen_p = self.settings.max_gen_p[balanced_id]
        min_gen_p = self.settings.min_gen_p[balanced_id]
        if obs is not None:
            dispatch = obs.actual_dispatch[balanced_id]
            has_overbalance = dispatch > max_gen_p or dispatch < min_gen_p
        return has_overbalance

    @property
    def has_emergency(self):
        return self.has_overflow or self.has_overbalance


class ActionMappingWrapper(Wrapper):
    def __init__(self, env, settings):
        """Map action space [-1, 1] of model output to new action space
        [low_bound, high_bound].
        """
        super(ActionMappingWrapper,self).__init__(env)
        self.settings = settings

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, model_output_act, **kwargs):
        """
        Args:
            model_output_act(np.array): The values must be in in [-1, 1].
        """
        action = action_process(self.settings, self.env.raw_obs, model_output_act)

        # return self.env.step(wrap_action(mapped_action),**kwargs)
        return self.env.step(action,**kwargs)

class HybridActionMappingWrapper(Wrapper):
    def __init__(self, env, settings):
        """Map action space [-1, 1] of model output to new action space
        [low_bound, high_bound].
        """
        super(HybridActionMappingWrapper,self).__init__(env)
        self.settings = settings

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, model_output_act, **kwargs):
        """
        Args:
            model_output_act(np.array): The values must be in in [-1, 1].
        """

        # gen_status = ((self.env.raw_obs.gen_status == 0) & (self.env.raw_obs.steps_to_recover_gen == 0)).astype(float)
        # gen_status = np.append(gen_status, 1.)
        # idx = np.where(gen_status==1)[0].tolist()
        # op = idx[np.random.randint(len(idx))]
        action = action_process(self.settings, self.env.raw_obs, model_output_act)

        # return self.env.step(wrap_action(mapped_action),**kwargs)
        return self.env.step(action,**kwargs)


class RewardWrapper(Wrapper):

    def __init__(self, env):
        super(RewardWrapper,self).__init__(env)
   
    def step(self, action, **kwargs):
        obs, reward, done, info = self.env.step(action, **kwargs)
        # training_reward = 1.0
        training_reward = reward
        if done and not info["timeout"]:
            training_reward -= 10.0
        info["origin_reward"] = reward
        return obs, training_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
    
def wrap_env(env, settings, obs_statistics=None):
    env = MaxTimestepWrapper(env)
    env = RewardWrapper(env)
    env = ObsTransformerWrapper(env, settings, obs_statistics)
    # env = ActionMappingWrapper(env, settings)
    env = HybridActionMappingWrapper(env, settings)
    return env