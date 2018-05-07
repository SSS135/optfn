import torch
import torch.autograd
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class GradRunningNorm(nn.Module):
    def __init__(self, weight=1, momentum=0.999):
        super().__init__()
        self.momentum = momentum
        self.weight = weight
        self.register_buffer('avg_sq', None)
        self.norm = torch.tensor(0)
        self.step = 0
        self.var_clone = VariableClone()
        self.var_clone.register_backward_hook(self.backward_hook)

    def backward_hook(self, module, grad_input, grad_output):
        assert len(grad_input) == 1, grad_input
        grad_input, = grad_input
        self.norm.lerp_(grad_input.detach().pow(2).sum(), 1 - self.momentum)
        self.step += 1
        bias_correction = 1 - self.momentum ** self.step
        norm = (self.norm / bias_correction).sqrt()
        return self.weight * grad_input / norm,

    def forward(self, input):
        return input.clone()


class VariableClone(nn.Module):
    def forward(self, input):
        return input.clone()
