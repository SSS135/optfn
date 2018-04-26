import torch
import torch.autograd
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class GradReweightFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, batch_weights):
        assert batch_weights.dim() == 1
        ctx.save_for_backward(batch_weights)
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        batch_weights, = ctx.saved_tensors
        xg = grad_output.view(grad_output.shape[0], -1)
        xg = xg / (xg.norm(2, dim=-1, keepdim=True) + 1e-5) * Variable(batch_weights.unsqueeze(-1))
        return xg.view_as(grad_output), None


def grad_reweight(x, batch_weights):
    return GradReweightFunction.apply(x, batch_weights)