import torch
import torch.nn as nn
import torch.nn.functional as F


class NonLocalBlock(nn.Module):
    def __init__(self, in_channels, reduction=4, conv=nn.Conv2d):
        super().__init__()
        self.reduction = reduction
        self.in_channels = in_channels
        self.conv_in = conv(in_channels, in_channels // reduction * 3, 1, bias=False)
        self.conv_out = conv(in_channels // reduction, in_channels, 1, bias=False)
        self.conv_out.weight.data.fill_(0)

    def forward(self, input):
        # B x C x HW
        emb_a, emb_b, g = self.conv_in(input).view(input.shape[0], self.conv_in.out_channels, -1).chunk(3, dim=1)
        # B x HW x HW
        weights = emb_a.transpose(1, 2) @ emb_b
        # B x HW x HW
        weights = F.softmax(weights, dim=2)
        # B x HW x C
        y = weights @ g.transpose(1, 2)
        # B x C x HW
        y = y.transpose(1, 2)
        # B x C x H x W
        y = y.view(*y.shape[:2], *input.shape[2:])
        y = self.conv_out(y)
        return input + y
