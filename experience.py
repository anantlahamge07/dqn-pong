from dataclasses import dataclass
import torch
import numpy as np
import typing as tt

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
