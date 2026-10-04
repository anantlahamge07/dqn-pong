
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
        pass

    def observation(self, obs: np.ndarry)