import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class GroupedLinear(nn.Module):
    def __init__(self, in_features, out_features, num_groups, bias=True):
        super().__init__()
        assert in_features % num_groups == 0 and out_features % num_groups == 0
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups
        self.weight = nn.Parameter(torch.Tensor(num_groups, in_features // num_groups, out_features // num_groups))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_groups, out_features // num_groups))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(-1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        assert input.dim() == 2
        B, C = input.shape
        NG, GCI, GCO = self.num_groups, self.in_features // self.num_groups, self.out_features // self.num_groups
        input = input.view(B, NG, 1, GCI)
        # (B, NG, 1, GCO)
        x = input @ self.weight
        x = x.view(B, NG * GCO)
        return x

    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}, ' \
               f'num_groups={self.num_groups}, bias={self.bias is not None}'
