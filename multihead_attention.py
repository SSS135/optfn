import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def multihead_attention(x, key_size, value_size):
    """

    Args:
        x: (batch, num_heads, key_size * 2 + value_size, num_keys)
        key_size: int
        value_size: int

    Returns: (batch, num_heads, value_size, num_keys)

    """
    KS, VS = key_size, value_size
    key, query, value = x.split([KS, KS, VS], dim=2)
    B, NH, _, NK = key.shape
    assert value.shape == (B, NH, VS, NK)
    # (B, NH, NK, NK)
    saliency = query.transpose(-1, -2) @ key / math.sqrt(KS)
    mask = F.softmax(saliency, dim=-1)
    assert mask.shape == (B, NH, NK, NK)
    # (B, NH, VS, NK)
    out = value @ mask.transpose(-1, -2)
    assert out.shape == (B, NH, VS, NK)
    return out


class MultiheadAttention2d(nn.Module):
    def __init__(self, num_heads, key_size, value_size, normalize=True):
        super().__init__()
        self.num_heads = num_heads
        self.key_size = key_size
        self.value_size = value_size
        self.normalize = normalize
        self.norm = nn.GroupNorm(3 * num_heads, num_heads * (2 * key_size + value_size))

    def forward(self, input):
        NH, KS, VS = self.num_heads, self.key_size, self.value_size
        x = self.norm(input) if self.normalize else input
        B, C, H, W = x.shape
        # (B, NH, KS * 2 + VS, H * W)
        x = x.view(B, NH, KS * 2 + VS, H * W)
        # (B, NH, KS, H * W)
        attn = multihead_attention(x, KS, VS)
        attn = attn.view(B, NH * VS, H, W)
        return attn


class MultiheadAttention1d(MultiheadAttention2d):
    def __init__(self, num_groups, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_groups = num_groups

    def forward(self, input):
        # (B, NH, KS * 2 + VS, NG)
        x = input.view(input.shape[0], self.num_heads, self.key_size * 2 + self.value_size, self.num_groups)
        # (B, NH, VS, NG)
        attn = super().forward(x)
        return attn.view(attn.shape[0], self.num_heads * self.value_size * self.num_groups)
