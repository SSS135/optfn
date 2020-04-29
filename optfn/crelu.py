import torch
import torch.nn as nn
import torch.nn.functional as F


class CReLU(nn.Module):
    def forward(self, x):
        return torch.cat([x.clamp(min=0), -x.clamp(max=0)], dim=1)

    @staticmethod
    def param_count_mul():
        return 2


class NCReLU(nn.Module):
    def forward(self, x):
        return torch.cat([x.clamp(min=0), x.clamp(max=0)], dim=1)

    @staticmethod
    def param_count_mul():
        return 2


class DELU(nn.Module):
    def forward(self, x):
        x = F.elu(x)
        return torch.cat([x.clamp(min=0), x.clamp(max=0)], dim=1)

    @staticmethod
    def param_count_mul():
        return 2


class NDELU(nn.Module):
    def forward(self, x):
        x = F.elu(x)
        return torch.cat([x.clamp(min=0), -x.clamp(max=0)], dim=1)

    @staticmethod
    def param_count_mul():
        return 2