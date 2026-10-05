import gymnasium as gym
import dqn_model
import wrappers
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
    torch.FloatTensor,        # rewards
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


