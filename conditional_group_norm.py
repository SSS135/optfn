import torch
import torch.nn as nn
from optfn.spectral_norm import spectral_init


class ConditionalGroupNorm(nn.Module):
    def __init__(self, num_groups, num_channels, condition_size, condition_fn, eps=1e-5):
        super().__init__()
        self.condition_fn = condition_fn
        self.norm = nn.GroupNorm(num_groups, num_channels, eps=eps, affine=False)
        self.embed = nn.Linear(condition_size, num_channels * 2)

    def forward(self, input):
        out = self.norm(input)
        embed = self.embed(self.condition_fn())
        shape = *input.shape[:2], *((input.dim() - 2) * [1])
        gamma, beta = [x.view(shape) for x in embed.chunk(2, 1)]
        out = gamma * out + beta
        return out