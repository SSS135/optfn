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
        self._avg_sq = 0
        self._step = 0
        self.register_backward_hook(self.backward_hook)

    def backward_hook(self, module, grad_input, grad_output):
        grad_input, = grad_input
        assert len(grad_input.shape) in (2, 3), grad_input.shape

        step_avg_sq = grad_input.detach().reshape(-1, grad_input.shape[-1]).pow(2).mean(0).sum().item() / (grad_input.shape[-1] ** 0.5)
        self._avg_sq = self.momentum * self._avg_sq + (1 - self.momentum) * step_avg_sq

        self._step += 1
        bias_correction = 1 - self.momentum ** self._step
        norm = (self._avg_sq / bias_correction) ** 0.5
        return self.weight / norm * grad_input,

    def forward(self, input: torch.Tensor):
        return input.clone()
