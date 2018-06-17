import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiheadAttention2d(nn.Module):
    def __init__(self, in_channels, num_heads, key_size, kernel_size=1, stride=1, padding=0, normalize=True):
        super().__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.key_size = key_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.normalize = normalize
        self.conv = nn.Conv2d(in_channels, 3 * num_heads * key_size, kernel_size, stride, padding, bias=not normalize)
        self.norm = nn.GroupNorm(3 * num_heads, 3 * num_heads * key_size)

    def forward(self, input):
        NH, KS = self.num_heads, self.key_size
        attn = self.conv(input)
        if self.normalize:
            attn = self.norm(attn)
        B, C, H, W = attn.shape
        # (B, NH, KS * 3, H * W)
        attn = attn.view(B, NH, KS * 3, H * W)
        # # (B, NH, H * W, KS + QS + VS)
        # attn = attn.permute(0, 1, 3, 2)
        # (B, NH, KS, H * W)
        key, query, value = attn.chunk(3, dim=2)
        assert value.shape == (B, NH, KS, H * W)
        # (B, NH, H * W, H * W)
        saliency = query.transpose(-1, -2) @ key / math.sqrt(KS)
        mask = F.softmax(saliency, dim=-1)
        assert mask.shape == (B, NH, H * W, H * W)
        # (B, NH, KS, H * W)
        out = value @ mask.transpose(-1, -2)
        assert out.shape == (B, NH, KS, H * W)
        # (B, NH, KS, H, W)
        # out = out.view(B, NH, H, W, KS).permute(0, 1, 4, 2, 3)
        out = out.view(B, NH * KS, H, W)
        return out
