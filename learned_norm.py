import torch
from torch.nn import Parameter, Module
from torch.autograd import Variable
from torch import nn


class LearnedNorm2d(Module):
    def __init__(self, num_features, reduction=8, eps=1e-2):
        super().__init__()
        self.num_features = num_features
        self.reduction = reduction
        self.eps = eps
        self.layers = nn.Sequential(
            nn.Linear(num_features * 2, num_features * 2 // reduction),
            nn.ReLU(True),
            nn.Linear(num_features * 2 // reduction, num_features * 2),
        )

    def _check_input_dim(self, input):
        if input.size(1) != self.num_features:
            raise ValueError('got {}-feature tensor, expected {}'
                             .format(input.size(1), self.num_features))

    def forward(self, input):
        self._check_input_dim(input)

        src_shape = input.shape
        b, c = input.size(0), input.size(1)
        input = input.contiguous().view(b, c, -1)

        mean, std = input.mean(-1), input.var(-1).add_(self.eps).sqrt_()
        net_mean, net_std = self.layers(torch.cat([mean, std], 1)).chunk(2, 1)
        std = std.log().add_(net_std).exp_().unsqueeze(-1)
        mean = net_mean.add(mean).unsqueeze(-1)

        input = input.sub(mean).div_(std).view(src_shape)
        return input

    def __repr__(self):
        return ('{name}({num_features}, eps={eps}, reduction={reduction})'
                .format(name=self.__class__.__name__, **self.__dict__))
