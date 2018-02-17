import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvAttention(nn.Module):
    def __init__(self, key_size, value_size):
        super().__init__()
        self.key_size = key_size
        self.value_size = value_size

    def forward(self, x):
        # vkq = input.split([self.value_size, self.key_size, self.key_size], 1)
        vkq = x[:, :self.value_size], x[:, self.value_size:self.value_size + self.key_size], x[:, self.value_size + self.key_size:]
        value, key, query = [x.contiguous().view(x.shape[0], x.shape[1], -1).transpose(1, 2).contiguous().view(-1, x.shape[1]) for x in vkq]
        weight = query @ key.t() / math.sqrt(self.key_size)
        attention = F.softmax(weight, 1) @ value
        out = attention.view(x.shape[0], -1, self.value_size).transpose(1, 2).contiguous().view(x.shape[0], self.value_size, *x.shape[2:])
        return out

