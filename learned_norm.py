import torch
from torch.nn import Parameter, Module
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F


class LearnedNorm2d(Module):
    def __init__(self, num_features, groups, reduction=8, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.groups = groups
        self.reduction = reduction
        self.eps = eps
        self.reg_loss = None
        self.layers = nn.Sequential(
            nn.Linear(num_features * 2, num_features // reduction),
            nn.ReLU(True),
            nn.Linear(num_features // reduction, num_features),
        )

    def _check_input_dim(self, input):
        if input.size(1) != self.num_features:
            raise ValueError('got {}-feature tensor, expected {}'
                             .format(input.size(1), self.num_features))

    def forward(self, input: Variable):
        self._check_input_dim(input)

        src_shape = input.shape
        b, c = input.size(0), input.size(1)

        input = input.contiguous().view(b, c, -1)

        var = input.var(-1).add_(self.eps)
        mean, log_var = input.mean(-1), var.log()
        bias = self.layers(torch.cat([mean, F.tanh(log_var / 5)], 1))

        g_mean = mean.view(b, self.groups, -1).mean(-1, keepdim=True)
        g_std = var.view(b, self.groups, -1).mean(-1, keepdim=True).sqrt_()
        g_input = input.view(b, self.groups, -1)
        g_normalized = g_input.sub(g_mean).div_(g_std)
        normalized = g_normalized.view(b, c, -1).add(bias.unsqueeze(-1))

        return normalized.view(src_shape)

    def __repr__(self):
        return ('{name}({num_features}, eps={eps}, reduction={reduction})'
                .format(name=self.__class__.__name__, **self.__dict__))
