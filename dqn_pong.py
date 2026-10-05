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
from experience import Experience
from experience import State
from experience import Action
from experience import BatchTensors

from torch.utils.tensorboard.writer import SummaryWriter

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
            state_tensor = state_tensor.unsqueeze(0)
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
            if episode_reward is None:
                episode_reward = self.total_reward
            else:
                episode_reward += self.total_reward
            self._reset()
        return episode_reward


def batch_to_tensor(batch: tt.List[Experience], device: torch.device) -> BatchTensors:
    states, actions, rewards, done_flags,  new_states = [], [], [], [], []
    for experience in batch:
        states.append(experience.state)
        actions.append(experience.action)
        rewards.append(experience.reward)
        done_flags.append(experience.done_trunc)
        new_states.append(experience.new_state)

    states_t = torch.as_tensor(np.asarray(states)).to(device)
    actions_t = torch.as_tensor(actions).to(device)
    rewards_t = torch.as_tensor(rewards).to(device)
    done_flags_t = torch.as_tensor(done_flags).to(device)
    new_states_t = torch.as_tensor(np.asarray(new_states)).to(device)
    return (states_t, actions_t, rewards_t, done_flags_t, new_states_t)

def calculate_loss(batch: tt.List[Experience], net: dqn_model.DQN, target_net: dqn_model.DQN, device: torch.device) -> torch.Tensor:
    # getting the batch as different tensors using batch_to_tensor() method
    states_t, actions_t, rewards_t, done_flags_t, new_states_t = batch_to_tensor(batch, device)
    # getting the Q values of the action taken
    # here we also used actions_t.unsqueeze(-1) here because the action_t has shape x for some value x, and gather expects (x,1) as the shape of actions_t
    state_action_values = net(states_t).gather(1, actions_t.unsqueeze(-1))

    # now we will disable the calculation of gradients
    with torch.no_grad():
        # getting the max q values for the next states
        next_state_values = target_net(new_states_t).max(1)[0]
        # setting the q values 0.0 for the episodes that has been ended or truncated
        next_state_values[done_flags_t] = 0.0
        next_state_values = next_state_values.detach()

    # the Bellman approximation
    expected_state_action_values = (hp.GAMMA * next_state_values + rewards_t).unsqueeze(-1)
    # returning the mean squared error loss
    return nn.MSELoss()(state_action_values, expected_state_action_values)




