import torch
import torch.autograd
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, Function


class DReLUFunction(Function):
    @staticmethod
    def forward(ctx, x, dim):
        ctx.save_for_backward(x)
        ctx.dim = dim
        a, b = x.chunk(2, dim)
        return a.clamp(min=0) - b.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        dim = ctx.dim
        a, b = x.chunk(2, dim)
        a_gin = Variable((a > 0).type_as(x)).mul_(grad_output)
        b_gin = Variable((b > 0).type_as(x).neg_()).mul_(grad_output)
        grad_input = torch.cat([a_gin, b_gin], dim)
        return grad_input, None


def drelu(x, dim=1):
    return DReLUFunction.apply(x, dim)


class DReLU(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return drelu(input, self.dim)