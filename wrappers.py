
import gymnasium as gym
import collections
import numpy as np
from stable_baselines3.common import atari_wrappers


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
    

# this wrapper just changes the shape of the observation from height, width, channel (HWC) to channel, height, width as required by Pytorch
class ImageToPytorch(gym.ObservationWrapper):

    def __init__(self, env: gym.Env):
        super(ImageToPytorch, self).__init__(env)
        obs = env.observation_space
        # asserting that it is an instance of Box
        assert isinstance(obs, gym.spaces.Box)
        # asserting that the shape of the observation has at most 3 elements (HWC)
        assert len(obs.shape) == 3
        new_shape = (obs.shape[-1], obs.shape[0], obs.shape[1])
        new_obs = gym.spaces.Box(
            obs.low.min(),
            obs.high.max(),
            shape = new_shape, 
            dtype = obs.dtype
        )
        self.observation_space = new_obs


    def observation(self, obs: np.ndarray):
        # putting the last index value on the front
        return np.moveaxis(obs, 2, 0)


def create_env(self, env_name: str):
    env = gym.make(env_name)
    env = atari_wrappers.AtariWrapper(env, clip_reward = False, noop_max = 0)
    env = ImageToPytorch(env)
    env = BufferWrapper(env, n_steps=4)
    return env