import torch
from torch.nn import Parameter, Module
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm


class _GroupNorm(_BatchNorm):
    def __init__(self, num_features, groups, eps=1e-5, affine=False):
        super().__init__(num_features, eps, 0.1, affine)
        assert num_features % groups == 0, (num_features, groups)
        self.groups = groups
        self.running_mean = torch.zeros(1)
        self.running_var = torch.ones(1)

    def forward(self, input: Variable):
        b, c = input.size(0), input.size(1)

        # Apply instance norm
        input_reshaped = input.transpose(0, 1).contiguous().view(c // self.groups, b * self.groups, *input.shape[2:])

        # Repeat stored stats and affine transform params
        running_mean = self.running_mean.repeat(input_reshaped.shape[1])
        running_var = self.running_var.repeat(input_reshaped.shape[1])

        out = F.batch_norm(
            input_reshaped, running_mean, running_var, None, None,
            True, self.momentum, self.eps)

        out = out.view(c, b, -1).transpose(0, 1)
        if self.affine:
            out = out.mul(self.weight.unsqueeze(-1)).add_(self.bias.unsqueeze(-1))

        return out.contiguous().view_as(input)


class GroupNorm1d(_GroupNorm):
    pass


class GroupNorm2d(_GroupNorm):
    pass