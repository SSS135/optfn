import torch
from torch.nn import Parameter, Module
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F


class BiasNorm(Module):
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        self.bias = Parameter(torch.Tensor(num_features))
        self.extra_loss = None
        self.extra_loss_name = 'bias_norm'
        self.reset_parameters()

    def reset_parameters(self):
        self.bias.data.fill_(0)

    def forward(self, input):
        wshape = -1, *([1] * (input.dim() - 2))
        bias = self.bias.view(wshape).expand_as(input)
        self.extra_loss = F.mse_loss(bias, -input.detach())
        return input + bias

    def extra_repr(self):
        return '{num_features}'.format(**self.__dict__)