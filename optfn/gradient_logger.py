import torch
from torch.autograd import Function
import numpy as np
from collections import defaultdict


class GradientLoggerFunction(Function):
    @staticmethod
    def forward(ctx, x, logger, tag, step):
        ctx.log_info = logger, tag, step
        return x

    @staticmethod
    def backward(ctx, grad_output):
        logger, tag, step = ctx.log_info
        logger.add_scalar(f'Gradient/Mean {tag}', grad_output.mean(), step)
        logger.add_scalar(f'Gradient/RMS {tag}', grad_output.pow(2).mean().sqrt(), step)
        logger.add_histogram(f'Gradient {tag}', grad_output, step)
        return grad_output, None, None, None


def log_gradients(x: torch.Tensor, logger, tag, step):
    return GradientLoggerFunction.apply(x, logger, tag, step) if x.requires_grad else x
