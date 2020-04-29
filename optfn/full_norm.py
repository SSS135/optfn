from torch.nn.modules.batchnorm import _BatchNorm
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import torch.nn as nn


class _FullNorm(_BatchNorm):
    def __init__(self, num_features, eps=1e-5):
        super().__init__(num_features, eps, momentum=0, affine=False)

    def forward(self, input):
        b, c = input.size(0), input.size(1)

        input_reshaped = input.transpose(0, 1).contiguous()

        running_mean = input_reshaped.data.new(b)
        running_std = input_reshaped.data.new(b)

        out = F.batch_norm(input_reshaped, running_mean, running_std, None, None, True, 0, self.eps)

        return out.transpose(0, 1).contiguous()


class FullNorm2d(_FullNorm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))
        super()._check_input_dim(input)
