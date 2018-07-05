import torch
import torch.nn as nn
import torch.nn.functional as F
from optfn.swish import swish


class ABU(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.full((5,), 1 / 5))

    def forward(self, input):
        activ = torch.stack([F.relu(input), F.elu(input), F.tanh(input), swish(input), input], -1)
        return activ.mul_(self.scale).sum(-1)
