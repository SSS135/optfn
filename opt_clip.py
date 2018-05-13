import torch
import torch.autograd
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class OptClipFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, pivot, clip):
        assert x.shape == pivot.shape and clip.dim() == 0
        ctx.save_for_backward(x, pivot, clip)
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        x, pivot, clip = ctx.saved_tensors
        grad_output = grad_output.clone()
        diff = x - pivot
        zero_mask = (diff > clip) & (grad_output < 0) | (diff < -clip) & (grad_output > 0)
        grad_output[zero_mask] = 0
        return grad_output, None, None


def opt_clip(x, pivot, clip):
    if not isinstance(clip, torch.Tensor):
        clip = torch.tensor(clip, dtype=x.dtype, device=x.device)
    return OptClipFunction.apply(x, pivot, clip)