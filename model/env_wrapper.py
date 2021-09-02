import gym
import numpy as np
from parl.utils import logger

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

    def __init__(self, env, settings):
        super(ObsTransformerWrapper,self).__init__(env)
        self.settings = settings
        self.has_overflow = False # Add an attribute to mark whether the env has overflowed lines.. 
        self.has_overbalance = False

    def _get_obs(self, obs):

        loads = []
        loads.append(obs.load_p)
        loads.append(obs.load_q)
        loads.append(obs.load_v)
        loads = np.concatenate(loads)

        # prods
        prods = []
        prods.append(obs.gen_p)
        prods.append(obs.gen_q)
        prods.append(obs.gen_v)
        prods = np.concatenate(prods)
        
        # rho
        rho = np.array(obs.rho) - 1.0

        next_load = obs.nextstep_load_p

        # action_space
        action_space_low = obs.action_space['adjust_gen_p'].low.tolist()
        action_space_high = obs.action_space['adjust_gen_p'].high.tolist()
        action_space_low[self.settings.balanced_id] = 0.0
        action_space_high[self.settings.balanced_id] = 0.0
        
        features = np.concatenate([
            loads, prods,
            rho.tolist(), next_load, action_space_low, action_space_high
        ])

        return features

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
        self.v_action = np.zeros(self.settings.num_gen)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, model_output_act, **kwargs):
        """
        Args:
            model_output_act(np.array): The values must be in in [-1, 1].
        """
        N = len(model_output_act)

        gen_p_action_space = self.env.raw_obs.action_space['adjust_gen_p']

        gen_p_low_bound = gen_p_action_space.low
        gen_p_high_bound = gen_p_action_space.high

        # gen_v_action_space = self.env.raw_obs.action_space['adjust_gen_v']

        # gen_v_low_bound = gen_v_action_space.low
        # gen_v_high_bound = gen_v_action_space.high

        # low_bound = np.concatenate([gen_p_low_bound, gen_v_low_bound])
        # high_bound = np.concatenate([gen_p_high_bound, gen_v_high_bound])
        low_bound = gen_p_low_bound
        high_bound = gen_p_high_bound
    
        mapped_action = low_bound + (model_output_act - (-1.0)) * (
            (high_bound - low_bound) / 2.0)
        mapped_action[self.settings.balanced_id] = 0.0
        # mapped_action[N//2 + self.settings.balanced_id] = 0.0
        mapped_action = np.clip(mapped_action, low_bound, high_bound)

        # return self.env.step(wrap_action(mapped_action),**kwargs)
        return self.env.step(form_action(mapped_action, self.v_action),**kwargs)

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
    
def wrap_action(action):
    N = len(action)
    act = {
        'adjust_gen_p': action[:N//2],
        'adjust_gen_v': action[N//2:]
    }
    return act
    
def form_action(adjust_gen_p, adjust_gen_v):
    return {'adjust_gen_p': adjust_gen_p, 'adjust_gen_v': adjust_gen_v}

def wrap_env(env, settings):
    env = MaxTimestepWrapper(env)
    env = RewardWrapper(env)
    env = ObsTransformerWrapper(env, settings)
    env = ActionMappingWrapper(env, settings)
    return env