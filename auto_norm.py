import torch
from torch.nn import Parameter, Module
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F


class AutoNorm(Module):
    def __init__(self, num_features, group_size=16, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.group_size = group_size
        self.weight = Parameter(torch.Tensor(4, num_features))
        self.bias = Parameter(torch.Tensor(4, num_features))
        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.normal_(1.0 / self.weight.shape[0], 0.01)
        self.bias.data.normal_(0, 1e-4)

    def forward(self, input):
        wshape = -1, *([1] * (input.dim() - 2))
        return F.group_norm(input, 1, self.weight[0], self.bias[0], self.eps) + \
               F.group_norm(input, max(1, self.num_features // self.group_size), self.weight[1], self.bias[1], self.eps) + \
               F.group_norm(input, self.num_features, self.weight[2], self.bias[2], self.eps) + \
               input * self.weight[3].view(wshape) + self.bias[3].view(wshape)

    def extra_repr(self):
        return '{num_features}, eps={eps}, group_size={group_size}'.format(**self.__dict__)