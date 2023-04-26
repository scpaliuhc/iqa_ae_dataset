import torch
import torch.nn as nn
class Normalize(nn.Module) :
    def __init__(self, mean, std,device) :
        super(Normalize, self).__init__()
        self.register_buffer('mean', torch.Tensor(mean).to(device))
        self.register_buffer('std', torch.Tensor(std).to(device))
    def forward(self, input):
        # Broadcasting
        mean = self.mean.reshape(1, 3, 1, 1)
        std = self.std.reshape(1, 3, 1, 1)
        return (input - mean) / std