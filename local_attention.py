import torch
import torch.nn as nn
import torch.nn.functional as F
from .tile_2d import Tile2d
import math


class LocalAttention2d(nn.Module):
    def __init__(self, in_channels, key_size, num_heads, kernel_size, stride=1, padding=0):
        super().__init__()
        self.in_channels = in_channels
        self.key_size = key_size
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.out_channels = key_size * num_heads
        self.attn_conv = nn.Conv2d(in_channels, key_size * 3 * num_heads, 1)
        self.tiler = Tile2d(self.attn_conv.out_channels, kernel_size, stride, padding)
        self.norm = nn.GroupNorm(3 * num_heads, key_size * 3 * num_heads)

    def forward(self, input):
        attn = self.attn_conv(input)
        # (B, C, K, K, OH, OW)
        tiles = self.tiler(attn)
        B, C, KH, KW, OH, OW = tiles.shape
        # (B, K, K, C, OH, OW)
        tiles = tiles.permute(0, 2, 3, 1, 4, 5).contiguous()
        assert tiles.shape == (B, KH, KW, C, OH, OW)
        tiles = self.norm(tiles.view(B * KH * KW, C, OH, OW)).view_as(tiles)
        # (B, OH, OW, C, K, K)
        tiles = tiles.permute(0, 4, 5, 3, 1, 2)
        assert tiles.shape == (B, OH, OW, C, KH, KW)
        # (B * OH * OW, NH, KS + QS + VS, K * K)
        VS, KS, NH, K = self.key_size, self.key_size, self.num_heads, self.kernel_size
        tiles = tiles.contiguous().view(B * OH * OW, NH, KS * 2 + VS, KH * KW)
        key, query, value = tiles.split([KS, KS, VS], dim=2)
        # (B * OH * OW, NH, KS, 1)
        query = query.mean(3, keepdim=True)
        # (B * OH * OW, NH, 1, K * K)
        saliency = query.transpose(-1, -2) @ key / math.sqrt(KS)
        assert saliency.shape == (B * OH * OW, NH, 1, K * K)
        # (B * OH * OW, NH, 1, K * K)
        mask = F.softmax(saliency, dim=-1)
        # (B * OH * OW, NH, VS, 1)
        out = value @ mask.transpose(-1, -2)
        assert out.shape == (B * OH * OW, NH, VS, 1)
        # (B, NH, VS, OH, OW)
        out = out.view(B, OH, OW, NH, VS).permute(0, 3, 4, 1, 2)
        # (B, NH * VS, OH, OW)
        out = out.view(B, NH * VS, OH, OW)
        return out.contiguous()

