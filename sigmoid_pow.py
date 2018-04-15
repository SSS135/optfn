import torch
import torch.autograd
import torch.nn as nn


class SigmoidPowFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, pow):
        ctx.save_for_backward(x)
        ctx.pow = pow
        return x.sigmoid().pow_(pow)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        pow = ctx.pow
        # a E^-x (1 + E^-x)^(-1 - a)
        nexp = torch.exp(-x)
        der = pow * nexp * (1 + nexp) ** (-1 - pow)
        return grad_output * torch.autograd.Variable(der), None


def sigmoid_pow(x, pow):
    return SigmoidPowFunction.apply(x, pow)


class SigmoidPow(nn.Module):
    def __init__(self, pow):
        super().__init__()
        self.pow = pow

    def forward(self, input):
        return sigmoid_pow(input, pow)