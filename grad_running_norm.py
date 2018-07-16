import torch
import torch.autograd
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class GradRunningNorm(nn.Module):
    def __init__(self, weight=1, momentum=0.99):
        super().__init__()
        self.momentum = momentum
        self.weight = weight
        self.avg_sq = 0
        self.step = 0
        self.register_backward_hook(self.backward_hook)

    def backward_hook(self, module, grad_input, grad_output):
        assert len(grad_input) == 1, grad_input
        grad_input, = grad_input
        self.avg_sq = self.momentum * self.avg_sq + (1 - self.momentum) * grad_input.detach().pow(2).sum().item()
        self.step += 1
        bias_correction = 1 - self.momentum ** self.step
        norm = (self.avg_sq / bias_correction) ** 0.5
        return grad_input * (self.weight / norm),

    def forward(self, input):
        return input.clone()
