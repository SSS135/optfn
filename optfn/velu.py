import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def velu(input, max_negative_value=0.1):
    a, b = input.view(input.shape[0], 2, -1).chunk(2, 1)
    r = F.relu(a)
    b = (b + math.log(max_negative_value)).clamp(-6, 6)
    m = a.clamp(min=0) + (b.exp() * ((a.clamp(max=0) * (-b).exp()).exp() - 1)).clamp(max=0)
    cat = torch.cat([r, m], 1)
    return cat.view_as(input)


class VELU(nn.Module):
    def __init__(self, max_negative_value=0.1):
        super().__init__()
        self.max_negative_value = max_negative_value

    def forward(self, input):
        return velu(input, self.max_negative_value)