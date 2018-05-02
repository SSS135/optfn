import torch.nn as nn


class TemporalGroupNorm(nn.GroupNorm):
    def forward(self, input):
        x = input.view(input.shape[0] * input.shape[1], input.shape[2])
        return super().forward(x).view_as(input)