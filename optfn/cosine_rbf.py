import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class CosRBF(nn.Module):
    def __init__(self, num_input, num_output, num_reduced):
        super().__init__()
        self.num_input = num_input
        self.num_output = num_output
        self.num_reduced = num_reduced
        self.reg_loss = None
        self.reductor = nn.Linear(num_input, num_reduced)
        self.centers = nn.Parameter(torch.Tensor(num_output, num_reduced).normal_())

    def forward(self, input):
        assert input.dim() == 2
        expand_shape = input.shape[0], *self.centers.shape
        reduced = self.reductor(input)
        reduced = reduced.unsqueeze(1).expand(expand_shape)
        centers = self.centers.unsqueeze(0).expand(expand_shape)
        self.reg_loss = F.cosine_similarity(reduced.detach(), centers, -1).abs() #+ (1 - (reduced.detach() - centers).std(-1)).abs()
        return F.cosine_similarity(reduced, centers.detach(), -1)