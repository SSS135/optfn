import torch
import torch.nn as nn
import math
import torch.autograd as autograd
import torch.nn.functional as F


def cosine_linear(input, weight, bias=None, wat=0.0001):
    w_norm = torch.sqrt(torch.sum(weight**2, dim=1, keepdim=True) + bias.unsqueeze(1)**2)
    x_norm = torch.sqrt(torch.sum(input**2, dim=1, keepdim=True) + wat**2)
    output = torch.matmul(input, weight.t()) + wat * bias
    # print('input', input.size(), 'weight', weight.size(), 'bias', bias.size(),
    #       'w_norm', w_norm.size(), 'x_norm', x_norm.size(), 'output', output.size())
    wx_normalized = output / (w_norm.sum() * x_norm)
    return wx_normalized


class CosineLinear(nn.Module):
    r"""Applies a linear transformation with cosine normalization to the incoming data

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to False, the layer will not learn an additive bias.
            Default: True

    Shape:
        - Input: :math:`(N, in\_features)`
        - Output: :math:`(N, out\_features)`

    Attributes:
        weight: the learnable weights of the module of shape
            (out_features x in_features)
        bias:   the learnable bias of the module of shape (out_features)

    Examples::

        >>> m = CosineLinear(20, 30)
        >>> input = autograd.Variable(torch.randn(128, 20))
        >>> output = m(input)
        >>> print(output.size())
    """

    def __init__(self, in_features, out_features, bias=True):
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        return cosine_linear(input, self.weight, self.bias)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'