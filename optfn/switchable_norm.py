import torch
from torch.nn import Parameter, Module
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F


class SwitchableNorm2d(Module):
    def __init__(self, num_features, eps=1e-5, affine=True, use_batch_norm=False, group_size=16):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.use_batch_norm = use_batch_norm
        self.group_size = group_size
        self.switch_weight = Parameter(torch.Tensor(2, 4))
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features))
            self.bias = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        self.switch_weight.data.fill_(0)
        if self.affine:
            self.weight.data.fill_(1)
            self.bias.data.zero_()

    def get_norm_stats(self, mean_inst, var_inst):
        mean_layer = mean_inst.mean(-1, keepdim=True)
        inst_sum = var_inst + mean_inst.pow(2)
        var_layer = inst_sum.mean(-1, keepdim=True) - mean_layer.pow(2)
        if self.use_batch_norm:
            mean_batch = mean_inst.mean(0, keepdim=True)
            var_batch = inst_sum.mean(0, keepdim=True) - mean_batch.pow(2)
        else:
            shape = mean_inst.shape[0], -1, self.group_size
            mean_batch, var_batch = mean_inst.view(*shape), var_inst.view(*shape)
            mean_batch, var_batch = [x.mean(-1, keepdim=True).expand_as(mean_batch).contiguous().view_as(mean_inst)
                                     for x in (mean_batch, var_batch)]
        return mean_layer, var_layer, mean_batch, var_batch

    def forward(self, input):
        assert input.dim() == 4

        x = input.view(*input.shape[:2], -1)
        mean_inst, var_inst = x.mean(-1), x.var(-1)
        mean_layer, var_layer, mean_batch, var_batch = self.get_norm_stats(mean_inst, var_inst)

        mean_inst_w, mean_layer_w, mean_batch_w, mean_off_w, var_inst_w, var_layer_w, var_batch_w, var_off_w = \
            torch.unbind(F.softmax(self.switch_weight, dim=-1).view(-1), dim=0)
        mean = mean_inst * mean_inst_w + mean_layer * mean_layer_w + mean_batch * mean_batch_w
        var = var_inst * var_inst_w + var_layer * var_layer_w + var_off_w + var_batch * var_batch_w + var_off_w.pow(2)
        std = var.add(self.eps).sqrt_()
        norm = (x - mean.unsqueeze(-1)) / std.unsqueeze(-1) * self.weight.unsqueeze(-1) + self.bias.unsqueeze(-1)
        return norm.view_as(input)

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}'.format(**self.__dict__)


class SwitchableNorm1d(SwitchableNorm2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.switch_weight = Parameter(torch.Tensor(2, 3))
        self.reset_parameters()

    def forward(self, input):
        assert input.dim() == 2

        mean_layer, var_layer, mean_batch, var_batch = self.get_norm_stats(input, input.pow(2))

        mean_layer_w, mean_batch_w, mean_off_w, var_layer_w, var_batch_w, var_off_w = \
            torch.unbind(F.softmax(self.switch_weight, dim=-1).view(-1), dim=0)
        mean = mean_layer * mean_layer_w + mean_batch * mean_batch_w
        var = var_layer * var_layer_w + var_batch * var_batch_w + var_off_w.pow(2)
        std = var.add(self.eps).sqrt_()
        norm = (input - mean) / std * self.weight + self.bias
        return norm