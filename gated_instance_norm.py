import torch
from torch.nn import Parameter, Module
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F


class GatedInstanceNorm2d(Module):
    def __init__(self, num_features, eps=1e-5, affine=True, init_gate=1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.init_gate = init_gate
        self.reg_loss = None
        self.gates_mean = nn.Parameter(torch.Tensor(num_features))
        self.gates_std = nn.Parameter(torch.Tensor(num_features))
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features))
            self.bias = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        self.gates_mean.data.fill_(self.init_gate)
        self.gates_std.data.fill_(self.init_gate)
        if self.affine:
            self.weight.data.fill_(1)
            self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.size(1) != self.num_features:
            raise ValueError('got {}-feature tensor, expected {}'
                             .format(input.size(1), self.num_features))

    def forward(self, input):
        self._check_input_dim(input)

        src_shape = input.shape
        b, c = input.size(0), input.size(1)
        input = input.contiguous().view(b, c, -1)

        self.reg_loss = 0.5 * self.gates_mean.mean() + 0.5 * self.gates_std.mean()

        mean, log_var = input.mean(-1), input.var(-1).add_(self.eps).log_()

        std = log_var.mul_(0.5).mul_(self.gates_std.sigmoid()).exp_()
        mean = mean.mul_(self.gates_mean.sigmoid())
        normalized = input.sub(mean.unsqueeze(-1)).mul_((self.weight / std).unsqueeze(-1)).add_(self.bias.unsqueeze(-1))

        return normalized.view(src_shape)

    def __repr__(self):
        return ('{name}({num_features}, eps={eps}, reduction={reduction})'
                .format(name=self.__class__.__name__, **self.__dict__))
