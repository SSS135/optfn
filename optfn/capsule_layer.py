import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .tile_2d import Tile2d
import math


def squash(tensor, dim=-1):
    squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1 + squared_norm)
    return scale * tensor / squared_norm.sqrt()


class Squash(nn.Module):
    def __init__(self, num_capsules):
        super().__init__()
        self.num_capsules = num_capsules

    def forward(self, input):
        b, c, h, w = input.shape
        return squash(input.view(b, self.num_capsules, -1, h, w), dim=2).view_as(input)


class CapsuleLayer(nn.Module):
    def __init__(self, in_caps, in_caps_size, out_caps, out_caps_size, kernel_size, stride=1, padding=0, num_iterations=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.in_caps = in_caps
        self.in_caps_size = in_caps_size
        self.out_caps = out_caps
        self.out_caps_size = out_caps_size
        self.num_iterations = num_iterations
        self.tile = Tile2d(in_caps * in_caps_size, kernel_size, stride, padding)
        self.weight = nn.Parameter(torch.randn(out_caps, in_caps * kernel_size ** 2, in_caps_size, out_caps_size))
        # self.bias = nn.Parameter(torch.zeros(out_caps, in_caps * kernel_size ** 2, out_caps_size))
        # self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.in_caps_size)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def route(self, x):
        # x - (B, INC 1152, ICS 8) -> (1, B, INC, 1, ICS)
        # w - (ONC 10, INC 1152, ICS 8, OCS 16) -> (ONC, 1, INC, ICS, OCS)
        # (ONC 10, B, INC 1152, 1, OCS 16)
        priors = x[None, :, :, None, :] @ self.weight[:, None, :, :, :]
        # priors += self.bias[:, None, :, None, :]

        logits = Variable(torch.zeros(*priors.size())).cuda()
        for i in range(self.num_iterations):
            probs = F.softmax(logits, dim=2)
            # (ONC, B, 1, 1, OCS)
            outputs = squash((probs * priors).sum(dim=2, keepdim=True))

            if i != self.num_iterations - 1:
                delta_logits = (priors * outputs).sum(dim=-1, keepdim=True)
                logits = logits + delta_logits

        ONC, B, _, _, OCS = outputs.shape
        # (B, ONC, OCS)
        outputs = outputs.view(ONC, B, OCS).permute(1, 0, 2).contiguous()
        return outputs

    def forward(self, input):
        # (B, C, K, K, OH, OW)
        tiles = self.tile(input)
        B, C, K, _, OH, OW = tiles.shape
        INC, ICS = self.in_caps, self.in_caps_size
        # (B * OH * OW, K * K, INC, ICS)
        tiles = tiles.view(B, INC, ICS, K * K, OH * OW).permute(0, 4, 3, 1, 2).contiguous()
        # (B * OH * OW, K * K * INC, ICS)
        tiles = tiles.view(B * OH * OW, K * K * INC, ICS)
        # (B * OH * OW, ONC, OCS)
        routed = self.route(tiles)
        ONC, OCS = self.out_caps, self.out_caps_size
        # (B, ONC * OCS, OH, OW)
        routed = routed.view(B, OH, OW, ONC * OCS).permute(0, 3, 1, 2).contiguous()
        return routed
