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


def leaky_tanh(x, leakage=0.2):
    return F.tanh(x) + leakage * x


def leaky_tanh_01(x, leakage=0.4, tanh_lim=(0.1, 0.9)):
    return (F.tanh(x).clamp(*tanh_lim) + leakage*x).clamp(0, 1)


def leaky_tanh_n11(x, leakage=0.1, tanh_lim=(-0.8, 0.8)):
    return (F.tanh(x).clamp(*tanh_lim) + leakage*x).clamp(-1, 1)


def bipow(x, power=0.9):
    return torch.pow(torch.abs(x), power)*torch.sign(x)


def selu(x):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * F.elu(x, alpha)


class LLLayer(nn.Module):
    def __init__(self, in_features, out_features, activation, bias=True, batchnorm=False):
        super(LLLayer, self).__init__()
        assert out_features % 2 == 0
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        if batchnorm:
            self.batchnorm = nn.BatchNorm1d(out_features)
        else:
            self.register_parameter('batchnorm', None)
        self.activation = activation

        init.orthogonal(self.linear.weight.data[:self.linear.out_features // 2])
        self.linear.weight.data[self.linear.out_features // 2:] = \
            -self.linear.weight.data[:self.linear.out_features // 2]
        self.linear.bias.data.fill_(0)

    def forward(self, x):
        lin = self.linear(x)
        x = self.activation(lin)
        if self.batchnorm:
            x = self.batchnorm(x)
        neg = -x[:, self.linear.out_features // 2:]
        pos = x[:, :self.linear.out_features // 2]
        x = torch.cat([pos, neg], 1)
        return x


class RandLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_bias=-3.0):
        super(RandLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_bias = sigma_bias
        self.mean_layer = nn.Linear(in_features, out_features)
        self.std_layer = nn.Linear(in_features, out_features)
        init.kaiming_uniform(self.mean_layer.weight)
        init.kaiming_uniform(self.std_layer.weight)
        self.std_layer.bias.data.fill_(0)
        self.std_layer.bias.data.fill_(sigma_bias)
        self._loss = None

    def forward(self, x):
        mean = self.mean_layer(x)
        if self.training:
            std = F.sigmoid(self.std_layer(x))
            self._loss = -std.mean()
            x = mean + std*Variable(torch.randn(mean.size()).type_as(mean.data))
            return x
        else:
            return mean

    @staticmethod
    def std_loss(model):
        losses = [m._loss for m in model.modules() if isinstance(m, RandLinear)]
        return torch.mean(torch.cat(losses))


class LLNetFixed(nn.Module):
    def __init__(self, n_in, n_out, activation=F.relu, layers=(64, 64, 64),
                 dropout=0.5, out_activation=F.log_softmax, cuda=True):
        super(LLNetFixed, self).__init__()
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.out_activation = out_activation
        self.layers = nn.ModuleList()
        for i in range(len(layers)):
            layer = nn.Linear(layers[i - 1] * 2 if i > 0 else n_in, layers[i])
            init.orthogonal(layer.weight.data)
            layer.bias.data.fill_(0)
            self.layers.append(layer)
        self.layers.append(nn.Linear(layers[-1] * 2, n_out))
        if cuda:
            self.cuda()

    def forward(self, x):
        for i in range(len(self.layers)):
            linear = self.layers[i]
            if i + 1 == len(self.layers):
                x = self.dropout(x)
            lin = linear(x)
            if i + 1 == len(self.layers):
                x = self.out_activation(lin)
            else:
                pos = self.activation(lin)
                neg = -self.activation(-lin)
                x = torch.cat([pos, neg], 1)
        return x


class LLNet(nn.Module):
    def __init__(self, n_in, n_out, activation=F.relu, layers=(64, 64, 64),
                 dropout=0.5, out_activation=F.log_softmax, cuda=True, batchnorm=False):
        super(LLNet, self).__init__()
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.out_activation = out_activation
        self.layers = nn.ModuleList()
        for i in range(len(layers)):
            layer = LLLayer(layers[i - 1] if i > 0 else n_in, layers[i], activation, batchnorm=batchnorm)
            self.layers.append(layer)
        self.layers.append(nn.Linear(layers[-1], n_out))
        if cuda:
            self.cuda()

    def forward(self, x):
        for i in range(len(self.layers)):
            layer = self.layers[i]
            x = layer(x)
            if i + 2 == len(self.layers):
                x = self.dropout(x)
            if i + 1 == len(self.layers):
                x = self.out_activation(x)
        return x


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


class CFCNet(nn.Module):
    def __init__(self, n_in, n_out, activation=F.elu, layers=(64, 64, 64),
                 dropout=0.5, out_activation=F.log_softmax, cuda=True, batchnorm=True, size_mult=64):
        super(CFCNet, self).__init__()
        self.n_in = n_in
        self.perm = torch.randperm(n_in*n_in)
        self.conv_seq = nn.Sequential(
            nn.Conv2d(1, size_mult, 5, 1, 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(size_mult),

            nn.Conv2d(size_mult, size_mult*2, 5, 1, 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(size_mult*2),

            nn.Conv2d(size_mult*2, size_mult*2, 5, 1, 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(size_mult*2),

            nn.Conv2d(size_mult*2, size_mult*4, 5, 1, 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(size_mult*4),

            nn.Conv2d(size_mult * 4, size_mult*4, 5, 1, 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(size_mult*4),

            nn.Conv2d(size_mult*4, size_mult, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(size_mult),

            nn.Conv2d(size_mult, n_out, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(n_out),
        )

    def forward(self, x):
        assert x.size(1) == self.n_in

        x = x.repeat(1, self.n_in)
        x = x.index_select(1, self.perm)
        x = x.view(x.size(0), 1, self.n_in, self.n_in)
        x = self.conv_seq(x)
        x = F.avg_pool2d(x, x.size(2))
        x = x.view(x.size(0), -1)
        return F.log_softmax(x)


class RandNet(nn.Module):
    def __init__(self, n_in, n_out, activation=F.elu, layers=(64, 64, 64), std_loss_mult=5,
                 dropout=0.5, out_activation=F.log_softmax, cuda=True, batchnorm=True):
        super(RandNet, self).__init__()
        self.activation = activation
        self.std_loss_mult = std_loss_mult
        self.dropout = nn.Dropout(dropout)
        self.out_activation = out_activation
        self.batchnorm = batchnorm
        self.linear_out = nn.Linear(layers[-1], n_out)
        self.layers = nn.ModuleList()
        if self.batchnorm:
            self.batchnorms = nn.ModuleList()
        else:
            self.register_parameter('batchnorms', None)
        for i in range(len(layers)):
            layer = RandLinear(layers[i - 1] if i > 0 else n_in, layers[i])
            self.layers.append(layer)
            if self.batchnorm:
                self.batchnorms.append(nn.BatchNorm1d(layer.out_features))
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

    def extra_loss(self):
        loss = self.std_loss_mult*RandLinear.std_loss(self)
        # print(loss.data[0])
        return loss


class DenseFCNet(nn.Module):
    def __init__(self, n_in, n_out, activation=F.elu, layers=(64, 64, 64),
                 dropout=0.5, out_activation=F.log_softmax, cuda=True, batchnorm=True):
        super(DenseFCNet, self).__init__()
        self.activation = activation
        self.dropout = dropout
        self.out_activation = out_activation
        self.linear_out = nn.Linear(layers[-1], n_out)
        self.layers = nn.ModuleList()
        self.batchnorm = batchnorm
        if self.batchnorm:
            self.batchnorms = nn.ModuleList()
        else:
            self.register_parameter('batchnorms', None)
        inputs = 0
        for i in range(len(layers)):
            inputs += layers[i - 1] if i > 0 else n_in
            layer = nn.Linear(inputs, layers[i])
            self.layers.append(layer)
            if self.batchnorm:
                self.batchnorms.append(nn.BatchNorm1d(layer.out_features))
        self.apply(weights_init)
        if cuda:
            self.cuda()

    def forward(self, x):
        for i in range(len(self.layers)):
            layer = self.layers[i]
            lin = layer(x)
            out = self.activation(lin)
            if self.batchnorm:
                out = self.batchnorms[i](out)
            x = torch.cat((x, out), dim=1) if i + 1 < len(self.layers) else out
        x = F.dropout(x, self.dropout, self.training)
        lin = self.linear_out(x)
        x = self.out_activation(lin)
        return x


class ConvNet(nn.Module):
    def __init__(self, feedforward_activation=F.elu, conv_activation=F.elu, out_activation=F.sigmoid, cuda=True,
                 layers=(nn.Conv2d(1, 10, 5),
                         nn.MaxPool2d(2),
                         nn.Conv2d(10, 20, 5),
                         nn.MaxPool2d(2),
                         nn.Linear(320, 50),
                         nn.Dropout(0.5),
                         nn.Linear(50, 10))):
        super(ConvNet, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.feedforward_activation = feedforward_activation
        self.conv_activation = conv_activation
        self.out_activation = out_activation
        self.apply(weights_init)
        if cuda:
            self.cuda()

    def forward(self, x):
        for i in range(len(self.layers)):
            layer = self.layers[i]
            lin = layer(x)

            activation = None
            if i + 1 == len(self.layers):
                activation = self.out_activation
            elif type(layer) is nn.Conv2d or type(layer) is nn.Linear:
                activation = self.conv_activation if type(layer) is nn.Conv2d else self.feedforward_activation

            x = activation(lin) if activation is not None else lin
        return x