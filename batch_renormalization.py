# https://discuss.pytorch.org/t/implementation-of-batch-renormalization-fails-unexpectedly-segementation-fault-core-dumped/1291

import torch
from torch.nn import Parameter, Module
from torch.autograd import Variable
import torch.nn.functional as F


def batch_renorm1d(input, running_mean, running_std, weight=None, bias=None,
                   training=False, momentum=0.01, eps=1e-5, rmax=3.0, dmax=5.0):
    if training:
        sample_mean = torch.mean(input, dim=0)
        sample_std = torch.std(input, dim=0) + eps

        r = torch.clamp(sample_std.data / running_std, 1. / rmax, rmax)
        d = torch.clamp((sample_mean.data - running_mean) / running_std, -dmax, dmax)

        input_normalized = (input - sample_mean) / sample_std * Variable(r) + Variable(d)

        running_mean += momentum * (sample_mean.data - running_mean)
        running_std += momentum * (sample_std.data - running_std)
    else:
        input_normalized = (input - running_mean) / running_std

    if weight is not None:
        return input_normalized * weight + bias
    else:
        return input_normalized


def batch_renorm2d(input, running_mean, running_std, weight=None, bias=None,
                   training=False, momentum=0.01, eps=1e-5, rmax=3.0, dmax=5.0):
    if training:
        # (C, B * H * W)
        input_1d = input.transpose(0, 1).contiguous().view(input.shape[1], -1)
        sample_mean = input_1d.mean(1)
        sample_std = (input_1d.var(1) + eps).sqrt()

        r = torch.clamp(sample_std.data / running_std, 1. / rmax, rmax)
        d = torch.clamp((sample_mean.data - running_mean) / running_std, -dmax, dmax)

        input_normalized = (input - sample_mean.view(1, -1, 1, 1)) / sample_std.view(1, -1, 1, 1)
        input_normalized = input_normalized * Variable(r.view(1, -1, 1, 1)) + Variable(d.view(1, -1, 1, 1))

        running_mean += momentum * (sample_mean.data - running_mean)
        running_std += momentum * (sample_std.data - running_std)
    else:
        input_normalized = (input - running_mean.view(1, -1, 1, 1)) / running_std.view(1, -1, 1, 1)

    if weight is not None:
        return input_normalized * weight.view(1, -1, 1, 1) + bias.view(1, -1, 1, 1)
    else:
        return input_normalized


class BatchReNorm1d(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, rmax=3.0, dmax=5.0, affine=True):
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
        return batch_renorm1d(input, self.running_mean, self.running_std, self.weight, self.bias,
                              self.training, self.momentum, self.eps, self.rmax, self.dmax)

    def __repr__(self):
        return ('{name}({num_features}, eps={eps}, momentum={momentum}, affine={affine})'
                .format(name=self.__class__.__name__, **self.__dict__))


class BatchReNorm2d(BatchReNorm1d):
    def forward(self, input):
        self._check_input_dim(input)
        return batch_renorm2d(input, self.running_mean, self.running_std, self.weight, self.bias,
                              self.training, self.momentum, self.eps, self.rmax, self.dmax)
