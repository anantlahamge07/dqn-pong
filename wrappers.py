
import gymnasium as gym
import collections
import numpy as np


class BufferWrapper(gym.ObservationWrapper):
    
    def __init__(self, env: gym.Env, n_steps):
        super(BufferWrapper, self).__init__(env)
        obs = env.observation_space
        # asserting whether the observation space is an instance of Box
        assert isinstance(obs, gym.spaces.Box)
        new_obs = gym.spaces.Box(
            obs.low.repeat(n_steps, axis = 0),
            obs.high.repeat(n_steps, axis = 0),
            dtype = obs.dtype
        )
        self.observation_space = new_obs
        self.buffer = collections.deque(maxlen=n_steps)

    def reset(self):
        # initially filling the buffer with the lowest possible values until the last value
        for _ in range(self.buffer.maxlen - 1):
            self.buffer.append(self.env.observation_space.low)
        obs, info = self.env.reset()
        return self.observation(obs), info

    def observation(self, obs: np.ndarray) -> np.ndarray:
        # appending the given observation to our buffer
        self.buffer.append(obs)
        # returning a concatenated array
        return np.concatenate(self.buffer)