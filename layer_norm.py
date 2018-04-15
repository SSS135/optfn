import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
import torch.nn.functional as F


class _LayerNorm(_BatchNorm):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__(num_features, eps, 0.1, affine)
        self.running_mean = torch.zeros(1)
        self.running_var = torch.ones(1)

    def forward(self, input):
        b, c = input.size(0), input.size(1)

        # Apply instance norm
        input_reshaped = input.transpose(0, 1).contiguous().view(c, -1)

        # Repeat stored stats and affine transform params
        running_mean = self.running_mean.repeat(input.shape[1])
        running_var = self.running_var.repeat(input.shape[1])

        out = F.batch_norm(
            input_reshaped, running_mean, running_var, None, None,
            not self.use_running_stats, self.momentum, self.eps)

        out = out.view(c, b, -1).transpose(0, 1)
        out = out.mul(self.weight.unsqueeze(-1)).add_(self.bias.unsqueeze(-1))

        return out.view_as(input)


class LayerNorm1d(_LayerNorm):
    pass


class LayerNorm2d(_LayerNorm):
    pass
