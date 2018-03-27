import torch
from torch.nn import Parameter, Module
from torch.autograd import Variable
from torch import nn


class LearnedNorm2d(Module):
    def __init__(self, num_features, reduction=8, eps=0.05):
        super().__init__()
        self.num_features = num_features
        self.reduction = reduction
        self.eps = eps
        self.reg_loss = None
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

        real_mean, real_std = input.mean(-1), input.var(-1).add_(self.eps).sqrt_()
        net_mean, net_logstd = self.layers(torch.cat([real_mean, real_std.sqrt().log()], 1)).chunk(2, 1)
        used_mean = real_mean + net_mean
        used_invstd = real_std.log().add(net_logstd).neg().exp()

        self.reg_loss = net_logstd.pow(2).mean() + net_mean.pow(2).mean()

        input = input.sub(used_mean.unsqueeze(-1)).mul_(used_invstd.unsqueeze(-1)).view(src_shape)
        return input

    def __repr__(self):
        return ('{name}({num_features}, eps={eps}, reduction={reduction})'
                .format(name=self.__class__.__name__, **self.__dict__))
