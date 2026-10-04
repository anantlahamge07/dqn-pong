import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()

        # Our model has been divided into 2 parts just for the sake of simplification
        # part 1: the convolution network
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride = 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
            )
        
        # just to get the input size for the linear network in the 2nd part
        size = self.conv(torch.zeros(1, *input_shape)).size()[-1]
        # part 2: the linear network
        self.fc = nn.Sequential(
            nn.Linear(size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    # we want the input to be a ByteTensor for the sake of memory efficiency and GPU bandwidth
    def forward(self, inputTensor: torch.ByteTensor):
        # normalizing the input and also converting the pixel values to floats
        # since our convolution network expects a float tensor
        x = inputTensor / 255.0
        return self.fc(self.conv(x))


        