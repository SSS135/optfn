import torch
import torch.nn as nn
import torch.nn.functional as F
from .tile_2d import Tile2d
import math
from torch.utils.checkpoint import checkpoint


class LocalAttention2d(nn.Module):
    def __init__(self, in_channels, num_heads, key_size, kernel_size, stride=1, padding=0,
                 conv_kernel_size=1, conv_stride=1, conv_padding=0):
        super().__init__()
        self.in_channels = in_channels
        self.key_size = key_size
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.out_channels = key_size * num_heads
        self.attn_conv = nn.Conv2d(in_channels, 3 * key_size * num_heads, conv_kernel_size, conv_stride, conv_padding)
        self.tiler = Tile2d(self.attn_conv.out_channels, kernel_size, stride, padding)
        self.norm = nn.GroupNorm(3 * num_heads, 3 * key_size * num_heads)

    def run_tiled(self, attn):
        # (B, C, K, K, OH, OW)
        tiles = self.tiler(attn)
        B, C, K, _, OH, OW = tiles.shape
        # (B, OH, OW, C, K, K)
        tiles = tiles.permute(0, 4, 5, 1, 2, 3)
        assert tiles.shape == (B, OH, OW, C, K, K)
        # (B * OH * OW, NH, KS + QS + VS, K * K)
        VS, KS, NH = self.key_size, self.key_size, self.num_heads
        tiles = tiles.contiguous().view(B * OH * OW, NH, KS * 2 + VS, K * K)
        # (B * OH * OW, NH, KS, K * K)
        key, query, value = tiles.split([KS, KS, VS], dim=2)
        # # (B * OH * OW, NH, KS, 1)
        # query = query.mean(3, keepdim=True)
        # (B * OH * OW, NH, 1, K * K)
        saliency = query.transpose(-1, -2) @ key / math.sqrt(KS)
        assert saliency.shape == (B * OH * OW, NH, K * K, K * K)
        # (B * OH * OW, NH, 1, K * K)
        mask = F.softmax(saliency, dim=-1)
        # (B * OH * OW, NH, VS, 1)
        out = value @ mask.transpose(-1, -2)
        assert out.shape == (B * OH * OW, NH, VS, K * K)
        # (B, NH, VS, OH, OW)
        out = out.mean(-1).view(B, OH, OW, NH, VS).permute(0, 3, 4, 1, 2)
        # (B, NH * VS, OH, OW)
        out = out.view(B, NH * VS, OH, OW)
        return out.contiguous()

    def forward(self, input):
        # (B, (KS + QS + VS) * NH, H, W)
        attn = self.attn_conv(input)
        attn = self.norm(attn)
        return checkpoint(self.run_tiled, attn) if attn.requires_grad else self.run_tiled(attn)


class AddLocationInfo2d(nn.Module):
    def __init__(self, config=((0.5, 0, 0), (1, 0, 0), (2, 0, 0), (4, 0, 0))):
        super().__init__()
        self.register_buffer('config', None)
        self.register_buffer('harr', None)
        self.register_buffer('warr', None)
        self.config = torch.tensor(config, dtype=torch.float32)
        self.harr = None
        self.warr = None

    def forward(self, input):
        with torch.no_grad():
            b, _, h, w = input.shape
            targs = dict(device=input.device, dtype=input.dtype)
            # if self.harr is None or self.harr.shape[2] != h or self.warr.shape[3] != w:
            harr = torch.arange(h, **targs).div_(h - 1).view(1, 1, h, 1)
            warr = torch.arange(w, **targs).div_(w - 1).view(1, 1, 1, w)
            scale, hoffset, woffset = [x.view(1, -1, 1, 1) for x in torch.unbind(self.config, -1)]
            harr, warr = [x.repeat(b, len(self.config), 1, 1).mul_(scale) for x in (harr, warr)]
            self.harr = harr.add_(hoffset).mul_(2 * math.pi)
            self.warr = warr.add_(woffset).mul_(2 * math.pi)
            # else:
            #     harr, warr = self.harr, self.warr
            #     scale = self.config[:, 0].view(1, -1, 1, 1)
            hrand, wrand = torch.empty((b, 2, 1, 1), **targs).uniform_(-1000, 1000).chunk(2, dim=1)
            loc = (harr + hrand).sin_() + (warr + wrand).sin_()
            loc.mul_(0.5)
        return torch.cat([input, loc], 1)


