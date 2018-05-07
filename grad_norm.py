import torch
import torch.autograd
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class GradNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, grad_mult):
        ctx.grad_mult = grad_mult
        ctx.mark_dirty(x)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_mult = ctx.grad_mult
        grad_output = grad_output * (grad_mult / grad_output.norm(2))
        return grad_output, None


def grad_norm(x, grad_mult=1):
    return GradNormFunction.apply(x, grad_mult)