import torch
import torch.nn as nn
import torch.nn.functional as F


def maxout(x, size, dim=1):
    assert dim == 1
    b, c, *tail = x.shape
    assert c % size == 0
    return x.contiguous().view(b, c // size, size, *tail).max(dim=2)[0]


class Maxout(nn.Module):
    def __init__(self, size, dim=1):
        super().__init__()
        assert dim == 1
        self.size = size
        self.dim = dim

    def forward(self, x):
        return maxout(x, self.size, self.dim)
