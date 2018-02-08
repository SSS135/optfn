# https://discuss.pytorch.org/t/implementation-of-batch-renormalization-fails-unexpectedly-segementation-fault-core-dumped/1291

import torch
from torch.nn import Parameter, Module
from torch.autograd import Variable


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
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('r', torch.ones(1))
        self.register_buffer('d', torch.zeros(1))
        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        self.r.fill_(1)
        self.d.zero_()
        if self.affine:
            self.weight.data.uniform_()
            self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.size(1) != self.running_mean.nelement():
            raise ValueError('got {}-feature tensor, expected {}'
                             .format(input.size(1), self.num_features))

    def forward(self, input):
        self._check_input_dim(input)

        if self.training:
            sample_mean = torch.mean(input, dim=0)
            sample_var = torch.var(input, dim=0).clamp(min=1e-4)

            self.r = torch.clamp(sample_var.data / self.running_var,
                                 1. / self.rmax, self.rmax)
            self.d = torch.clamp((sample_mean.data - self.running_mean) / self.running_var,
                                 -self.dmax, self.dmax)

            input_normalized = (input - sample_mean) / sample_var * Variable(self.r) + Variable(self.d)

            self.running_mean += self.momentum * (sample_mean.data - self.running_mean)
            self.running_var += self.momentum * (sample_var.data - self.running_var)
        else:
            input_normalized = (input - self.running_mean) / self.running_var

        if self.affine:
            return input_normalized * self.weight + self.bias
        else:
            return input_normalized

    def __repr__(self):
        return ('{name}({num_features}, eps={eps}, momentum={momentum}, affine={affine})'
                .format(name=self.__class__.__name__, **self.__dict__))


class BatchReNorm2d(BatchReNorm1d):
    def forward(self, input):
        s = input.shape
        x = input.transpose(1, -1).contiguous().view(-1, s[1]).contiguous()
        x = super().forward(x)
        return x.view(s[0], -1, s[1]).transpose(1, -1).contiguous().view_as(input).contiguous()
