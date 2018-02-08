import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.autograd import Variable
from torch.nn import init
import torch.autograd


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.kaiming_uniform(m.weight.data)
        m.bias.data.fill_(0)
    if isinstance(m, nn.Linear):
        init.kaiming_uniform(m.weight.data)
        m.bias.data.fill_(0)


class LadderNetBase(nn.Module):
    def __init__(self):
        super(LadderNetBase, self).__init__()


class FCNet(nn.Module):
    def __init__(self, n_in, n_out, activation=F.elu, layers=(64, 64, 64),
                 dropout=0.5, out_activation=F.log_softmax, cuda=True, batchnorm=True):
        super(FCNet, self).__init__()
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.out_activation = out_activation
        self.batchnorm = batchnorm
        self.linear_out = nn.Linear(layers[-1], n_out)
        self.layers = nn.ModuleList()
        self.batchnorms = nn.ModuleList()
        if self.batchnorm:
            self.batchnorms = nn.ModuleList()
        else:
            self.register_parameter('batchnorms', None)
        for i in range(len(layers)):
            layer = nn.Linear(layers[i - 1] if i > 0 else n_in, layers[i])
            self.layers.append(layer)
            if self.batchnorm:
                self.batchnorms.append(nn.BatchNorm1d(layer.out_features))
        self.apply(weights_init)
        if cuda:
            self.cuda()

    def forward(self, x):
        for i in range(len(self.layers)):
            lin = self.layers[i](x)
            x = self.activation(lin)
            if self.batchnorm:
                x = self.batchnorms[i](x)
        x = self.dropout(x)
        lin = self.linear_out(x)
        x = self.out_activation(lin)
        return x