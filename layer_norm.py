import torch
import torch.nn as nn
from optfn.group_norm import _GroupNorm
import torch.nn.functional as F


class _LayerNorm(_GroupNorm):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__(num_features, 1, eps, affine)


class LayerNorm1d(_LayerNorm):
    pass


class LayerNorm2d(_LayerNorm):
    pass
