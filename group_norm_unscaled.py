import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupNormUnscaled(nn.GroupNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.affine:
            weight = self.weight.requires_grad_(False).fill_(1)
            del self.weight
            self.register_buffer('weight', weight)