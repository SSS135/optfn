import torch
from torch.autograd import Function
import numpy as np
from collections import defaultdict


grad_mean = defaultdict(list)
grad_square = defaultdict(list)
log_interval = 100


class GradientLoggerFunction(Function):
    @staticmethod
    def forward(ctx, x, key):
        ctx.key = key
        return x

    @staticmethod
    def backward(ctx, grad_output):
        mean = grad_mean[ctx.key]
        square = grad_square[ctx.key]
        mean.append(grad_output.mean().item())
        square.append(grad_output.pow(2).mean().item())
        if len(mean) >= log_interval:
            std = (np.array(square) - np.array(mean) ** 2) ** 0.5
            print(f'{ctx.key} grad rms {np.mean(square) ** 0.5:.6}, mean {np.mean(mean):.6}, std {np.mean(std):.6}')
            mean.clear()
            square.clear()
        return grad_output, None


def log_gradients(x: torch.Tensor, key: str):
    return GradientLoggerFunction.apply(x, key) if x.requires_grad else x
