import torch
import torch.nn as nn
from torch.autograd import Variable


class ShuffleConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias)
        self.register_buffer('permutation', torch.randperm(out_channels))

    def forward(self, input):
        x = super().forward(input)
        perm = Variable(self.permutation)
        return x[:, perm].contiguous()
