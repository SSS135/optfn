import torch
import torch.nn as nn


def drrelu(x, neg_min=-0.3, neg_max=0.3, pos_min=0.95, pos_max=1.05, inplace=False):
    if not inplace:
        x = x.clone()
    with torch.no_grad():
        neg_mask = x < 0
        pos_mask = x > 0
        rand = torch.rand_like(x)
        rand[neg_mask] = rand[neg_mask].mul_(neg_max - neg_min).add_(neg_min)
        rand[pos_mask] = rand[pos_mask].mul_(pos_max - pos_min).add_(pos_min)
    x *= rand
    # x[neg_mask] *= rand[neg_mask] * (neg_max - neg_min) + neg_min
    # x[pos_mask] *= rand[pos_mask] * (pos_max - pos_min) + pos_min
    return x


class DRReLU(nn.Module):
    def __init__(self, neg_min, neg_max, pos_min, pos_max, inplace=False):
        super().__init__()
        self.neg_min = neg_min
        self.neg_max = neg_max
        self.pos_min = pos_min
        self.pos_max = pos_max
        self.inplace = inplace

    def forward(self, input):
        return drrelu(input, self.neg_min, self.neg_max, self.pos_min, self.pos_max)