import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PlasticLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, single_plastic_lr=False, initial_plastic_lr=0.1, oja_rule=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.single_plastic_lr = single_plastic_lr
        self.initial_plastic_lr = initial_plastic_lr
        self.oja_rule = oja_rule
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.plastic_scale = nn.Parameter(torch.Tensor(out_features, in_features))
        self.plastic_lr = nn.Parameter(torch.Tensor(1) if single_plastic_lr else torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.plastic_scale.data.uniform_(-stdv, stdv)
        if self.single_plastic_lr:
            self.plastic_lr.data.fill_(self.initial_plastic_lr)
        else:
            self.plastic_lr.data.uniform_(min(self.initial_plastic_lr, 1e-6), self.initial_plastic_lr)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input: torch.Tensor, hebb: torch.Tensor):
        if hebb is None:
            hebb = input.new_zeros((input.shape[0], self.out_features, self.in_features))
        out = input.unsqueeze(-2) @ (self.weight.unsqueeze(0) + self.plastic_scale.unsqueeze(0) * hebb).transpose(-1, -2)
        out = out.squeeze(-2)
        uin, uout = input.unsqueeze(-2), out.unsqueeze(-1)
        if self.oja_rule:
            hebb = hebb + self.plastic_lr * uout * (uin - uout * hebb)
        else:
            hebb = self.plastic_lr * uin * uout + (1 - self.plastic_lr) * hebb
        if self.bias is not None:
            out = out + self.bias
        return out, hebb


class PlasticLinearRec(nn.Module):
    def __init__(self, num_features, single_plastic_lr=True, initial_plastic_lr=0.01, oja_rule=True):
        super().__init__()
        self.num_features = num_features
        self.single_plastic_lr = single_plastic_lr
        self.initial_plastic_lr = initial_plastic_lr
        self.oja_rule = oja_rule
        self.weight = nn.Parameter(torch.Tensor(num_features, num_features))
        self.plastic_scale = nn.Parameter(torch.Tensor(num_features, num_features))
        self.plastic_lr = nn.Parameter(torch.Tensor(1) if single_plastic_lr else torch.Tensor(num_features, num_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.plastic_scale.data.uniform_(-stdv, stdv)
        self.plastic_scale.data -= torch.diag(torch.diag(self.plastic_scale.data))
        if self.single_plastic_lr:
            self.plastic_lr.data.fill_(self.initial_plastic_lr)
        else:
            self.plastic_lr.data.uniform_(min(self.initial_plastic_lr, 1e-6), self.initial_plastic_lr)

    def forward(self, input: torch.Tensor, memory):
        if memory is None:
            last_out, hebb = input.new_zeros((input.shape[0], self.num_features)), \
                             input.new_zeros((input.shape[0], self.num_features, self.num_features))
        else:
            last_out, hebb = memory
        out = last_out.unsqueeze(-2) @ (self.weight.unsqueeze(0) + self.plastic_scale.unsqueeze(0) * hebb).transpose(-1, -2)
        out = F.tanh(out.squeeze(-2) + input)
        uin, uout = last_out.unsqueeze(-2), out.unsqueeze(-1)
        if self.oja_rule:
            hebb = hebb + self.plastic_lr * uout * (uin - uout * hebb)
        else:
            hebb = self.plastic_lr * uin * uout + (1 - self.plastic_lr) * hebb
        return out, (out, hebb)