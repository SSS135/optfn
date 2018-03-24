# https://discuss.pytorch.org/t/implementation-of-batch-renormalization-fails-unexpectedly-segementation-fault-core-dumped/1291

import torch
from torch.nn import Parameter, Module
from torch.autograd import Variable


def batch_renorm(input, running_mean, running_std, weight=None, bias=None,
                 training=False, momentum=0.01, eps=1e-5, rmax=3.0, dmax=5.0):
    b, c, _, _ = input.shape
    input_3d = input.view(b, c, -1)

    if training:
        # (C, B * H * W)
        input_1d = input_3d.transpose(0, 1).contiguous().view(c, -1)
        sample_mean = input_1d.mean(1)
        sample_std = input_1d.var(1).add_(eps).sqrt_()

        r = (sample_std.data / running_std).clamp_(1. / rmax, rmax)
        d = (sample_mean.data - running_mean).div_(running_std).clamp_(-dmax, dmax)

        input_normalized = (input_3d - sample_mean.view(1, -1, 1)).div_(sample_std.view(1, -1, 1))
        input_normalized.mul_(Variable(r.view(1, -1, 1))).add_(Variable(d.view(1, -1, 1)))

        running_mean.lerp_(sample_mean.data, momentum)
        running_std.lerp_(sample_std.data, momentum)
    else:
        input_normalized = (input_3d - Variable(running_mean.view(1, -1, 1))).div_(Variable(running_std.view(1, -1, 1)))

    if weight is not None:
        input_normalized.mul_(weight.view(1, -1, 1)).add_(bias.view(1, -1, 1))

    return input_normalized.view_as(input)


class BatchReNorm1d(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, rmax=3.0, dmax=5.0):
        super(BatchReNorm1d, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        self.momentum = momentum
        self.rmax = rmax
        self.dmax = dmax
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features))
            self.bias = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_std', torch.ones(num_features))
        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_std.fill_(1)
        if self.affine:
            self.weight.data.uniform_(0.95, 1.05)
            self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.size(1) != self.running_mean.nelement():
            raise ValueError('got {}-feature tensor, expected {}'
                             .format(input.size(1), self.num_features))

    def forward(self, input):
        self._check_input_dim(input)
        return batch_renorm(input, self.running_mean, self.running_std, self.weight, self.bias,
                            self.training, self.momentum, self.eps, self.rmax, self.dmax)

    def __repr__(self):
        return ('{name}({num_features}, eps={eps}, momentum={momentum}, affine={affine})'
                .format(name=self.__class__.__name__, **self.__dict__))


class BatchReNorm2d(BatchReNorm1d):
    pass
