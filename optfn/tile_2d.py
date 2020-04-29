import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Tile2d(nn.Module):
    """
    Split (B, C, IH, IW) feature map to (B, C, K, K, OH, OW)
    """
    def __init__(self, channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.register_buffer('weight', None)
        self.weight = torch.zeros(channels, kernel_size ** 2, 1, kernel_size, kernel_size)
        w = torch.zeros(kernel_size ** 2)
        for i in range(kernel_size ** 2):
            w.zero_()
            w[i] = 1
            self.weight[:, i, 0] = \
                w.view(1, kernel_size, kernel_size).expand(channels, kernel_size, kernel_size)
        self.weight = self.weight.view(channels * kernel_size ** 2, 1, kernel_size, kernel_size)

    def forward(self, input):
        assert input.shape[1] == self.channels
        x = F.conv2d(input, self.weight, bias=None,
                     stride=self.stride, padding=self.padding, groups=self.channels)
        x = x.view(x.shape[0], self.channels, self.kernel_size, self.kernel_size, *x.shape[-2:])
        # x = x.permute(0, 4, 5, 1, 2, 3).contiguous()
        # x = x.view(-1, *x.shape[2:])
        return x