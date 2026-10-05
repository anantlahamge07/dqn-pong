import gymnasium as gym
import dqn_model
import wrappers
import ExperienceBuffer
import hyperparameters as hp

from dataclasses import dataclass
import argparse
import time
import numpy as np
import collections
import typing as tt
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard.writer import SummaryWriter


# here we will define now type allies
State = np.ndarray
Action = int
BatchTensors = tt.Tuple[
    torch.ByteTensor,    # current state
    torch.LongTensor,    # actions
    torch.FloatTensor,   # rewards
    torch.BoolTensor,    # done || truncated
    torch.ByteTensor     # next state
    ]

# this will be used to keep entries in the experience replay buffer
@dataclass
class Experience:
    state: State
    action: Action
    reward: float
    done_trunc: bool
    new_state: State



class Agent:
    def __init__(self, env: gym.Env, exp_buffer: ExperienceBuffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self.state: tt.Optional[np.ndarray] = None
        self._reset()

    def _reset(self):
        self.state, _ = self.env.reset()
        self.total_reward = 0.0


    """ this decorator tells PyTorch to not calculate the gradients saving a lot of memory and computation, since 
        we are just asking the NN to make prediction for the given state """
    @torch.no_grad()
    def _step(self, net: dqn_model.DQN, device: torch.device, epsilon: float) -> tt.Optional[float]:
        
        # reward accumulated over the whole episode
        episode_reward = None
        if np.random.random() < epsilon:
            # choosing a random action (exploration)
            action = self.env.action_space.sample()
        else: # (exploitation using the neural network)
            # converting the current state (ndarray) to a tensor and putting it on the device, where our NN is present
            state_tensor = torch.as_tensor(self.state).to(device)
            # adding a batch dimension at position 0
            state_tensor.unsqueeze(0)
            # passing the tensor to our NN
            q_values = net(state_tensor)
            # getting the index(action itself) of the maximum q value
            _ , index_tensor = torch.max(q_values, dim = 1)
            # extracting the number from the tensor and converting to an integer
            action = int(index_tensor.item())
        

        # taking a step in th environment
        next_state, reward, is_done, is_truncated, _ = self.env.step(action)
        self.total_reward += reward
        # creating an Experience instance
        exp = Experience(
            self.state,
            action,
            reward,
            is_done or is_truncated,
            next_state
        )
        # appending the obtained experience to our experience 
        self.exp_buffer.append(exp)
        # updating the current state
        self.state = next_state
        if is_done or is_truncated:
            episode_reward += self.total_reward
            self._reset()
        return episode_reward


def batch_to_tensor(batch: tt.List[Experience], device: torch.device) -> BatchTensors:
    pass

def calculate_loss():
    pass


if __name__ == "__main__":
    pass