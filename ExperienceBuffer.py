import collections
from dqn_pong import Experience
import typing as tt
import numpy as np

class ExperienceBuffer:
    def __init__(self, capacity: int):
        self.buffer = collections.deque(maxlen=capacity)
        
    #  returning the length of the buffer
    def len(self):
        return len(self.buffer)
    
    # appending a new experience(a new transition) to our buffer
    def append(self, experience: Experience):
        self.buffer.append(experience)

    # sampling few random transitions from our experience replay buffer for training
    def sample(self, batch_size: int) -> tt.List[Experience]:
        random_indices = np.random.choice(self.len(), batch_size, replace=False)
        return [self.buffer[i] for i in random_indices]
        