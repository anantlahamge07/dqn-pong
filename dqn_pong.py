import gymnasium as gym
import dqn_model
import wrappers

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

# defining the hyperparameters
DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
# reward bound for the last 100 episodes to stop training
MEAN_REWARD_BOUND = 19

# γ used for the bellman approximation
GAMMA = 0.99
# the batch size sampled from the replay buffer
BATCH_SIZE = 32
# the maximum size of the replay buffer
REPLAY_SIZE = 10000
# the learning rate used in the adam optimizer(for our case adam)
LEARNING_RATE = 1e-4
# the count of frames we wait for before starting training to populate the replay buffer 
REPLAY_START_SIZE = 10000
# how frequently we sync model weights from the training model to the target model, which is used to get the value of the next state in the Bellman approximation
SYNC_TARGET_FRAMES = 1000

# for the epsilon decay schedule
EPSILON_DECAY_LAST_FRAME = 150000
# we start with epsilon = 1.0, which means full exploration in the early stages of training
EPSILON_START = 1.0
# epsilon will be linearly decayed to 0.01 in the first 150000 frames
EPSILON_FINAL = 0.01

