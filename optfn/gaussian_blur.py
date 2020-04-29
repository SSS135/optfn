import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.stats


def gaussian_kernel(kernel_size=21, sigma=3):
    assert kernel_size % 2 == 1
    x = np.linspace(-sigma, sigma, kernel_size)
    kernel_1d = scipy.stats.norm.pdf(x)
    kernel = np.sqrt(np.outer(kernel_1d, kernel_1d))
    kernel = kernel / kernel.sum()
    return kernel


class GaussianBlur2d(nn.Module):
    def __init__(self, num_channels, kernel_size, sigma, stride=1, padding=0):
        super().__init__()
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.stride = stride
        self.padding = padding
        self.register_buffer('weight', None)
        self.weight = torch.tensor(gaussian_kernel(kernel_size, sigma), dtype=torch.float)
        self.weight = self.weight.expand(num_channels, 1, kernel_size, kernel_size)

    def forward(self, input):
        return F.conv2d(input, self.weight, stride=self.stride, padding=self.padding, groups=self.num_channels)