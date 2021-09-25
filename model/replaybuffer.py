import numpy as np
from parl.utils import logger

class EnsembleReplayMemory(object):
    def __init__(self, max_size, obs_dim, act_dim, num_ensemble):
        self.max_size = int(max_size)
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.obs = np.zeros((max_size, obs_dim), dtype='float32')
        if act_dim == 0:  # Discrete control environment
            self.action = np.zeros((max_size, ), dtype='int32')
        else:  # Continuous control environment
            self.action = np.zeros((max_size, act_dim), dtype='float32')
        self.reward = np.zeros((max_size, ), dtype='float32')
        self.terminal = np.zeros((max_size, ), dtype='bool')
        self.next_obs = np.zeros((max_size, obs_dim), dtype='float32')
        self.mask = np.zeros((max_size, num_ensemble), dtype='bool')

        self._curr_size = 0
        self._curr_pos = 0

    def sample_batch(self, batch_size):
        batch_idx = np.random.randint(self._curr_size, size=batch_size)

        obs = self.obs[batch_idx]
        reward = self.reward[batch_idx]
        action = self.action[batch_idx]
        next_obs = self.next_obs[batch_idx]
        terminal = self.terminal[batch_idx]
        mask = self.mask[batch_idx]
        return obs, action, reward, next_obs, terminal, mask

    def make_index(self, batch_size):
        batch_idx = np.random.randint(self._curr_size, size=batch_size)
        return batch_idx

    def sample_batch_by_index(self, batch_idx):
        obs = self.obs[batch_idx]
        reward = self.reward[batch_idx]
        action = self.action[batch_idx]
        next_obs = self.next_obs[batch_idx]
        terminal = self.terminal[batch_idx]
        mask = self.mask[batch_idx]
        return obs, action, reward, next_obs, terminal, mask

    def append(self, obs, act, reward, next_obs, terminal, mask):
        if self._curr_size < self.max_size:
            self._curr_size += 1
        self.obs[self._curr_pos] = obs
        self.action[self._curr_pos] = act
        self.reward[self._curr_pos] = reward
        self.next_obs[self._curr_pos] = next_obs
        self.terminal[self._curr_pos] = terminal
        self.mask[self._curr_pos] = mask
        self._curr_pos = (self._curr_pos + 1) % self.max_size

    def size(self):
        return self._curr_size

    def __len__(self):
        return self._curr_size

    def save(self, pathname):
        other = np.array([self._curr_size, self._curr_pos], dtype=np.int32)
        np.savez(
            pathname,
            obs=self.obs,
            action=self.action,
            reward=self.reward,
            terminal=self.terminal,
            next_obs=self.next_obs,
            mask=self.mask,
            other=other)

    def load(self, pathname):
        data = np.load(pathname)
        other = data['other']
        if int(other[0]) > self.max_size:
            logger.warn('loading from a bigger size rpm!')
        self._curr_size = min(int(other[0]), self.max_size)
        self._curr_pos = min(int(other[1]), self.max_size - 1)

        self.obs[:self._curr_size] = data['obs'][:self._curr_size]
        self.action[:self._curr_size] = data['action'][:self._curr_size]
        self.reward[:self._curr_size] = data['reward'][:self._curr_size]
        self.terminal[:self._curr_size] = data['terminal'][:self._curr_size]
        self.next_obs[:self._curr_size] = data['next_obs'][:self._curr_size]
        self.mask[:self._curr_size] = data['mask'][:self._curr_size]
        logger.info("[load rpm]memory loade from {}".format(pathname))