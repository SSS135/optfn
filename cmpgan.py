import torch
import torch.nn as nn
import torch.nn.functional as F
from optfn.spectral_norm import spectral_norm
from optfn.drrelu import DRReLU
from optfn.multihead_attention import AdditiveMultiheadAttention2d
from optfn.group_norm_unscaled import GroupNormUnscaled


def spectral_init(module, gain=1):
    nn.init.kaiming_uniform_(module.weight, gain)
    if module.bias is not None:
        module.bias.data.zero_()
    return spectral_norm(module)


class GanD(nn.Module):
    def __init__(self, nc, nf):
        super().__init__()
        ng = 2
        self.net = nn.Sequential(
            spectral_init(nn.Conv2d(nc, nf, 4, 2, 1)),
            # GroupNormUnscaled(ng, nf),
            nn.LeakyReLU(0.2, True),
            spectral_init(nn.Conv2d(nf, nf * 2, 4, 2, 1, bias=False)),
            GroupNormUnscaled(ng * 2, nf * 2),
            nn.LeakyReLU(0.2, True),
            AdditiveMultiheadAttention2d(nf * 2, 1, nf * 2 // 8, normalize=False),
            spectral_init(nn.Conv2d(nf * 2, nf * 4, 4, 2, 1, bias=False)),
            GroupNormUnscaled(ng * 4, nf * 4),
            nn.LeakyReLU(0.2, True),
            spectral_init(nn.Conv2d(nf * 4, 1, 4, 1, 0, bias=False)),
        )

    def forward(self, input):
        return self.net(input).view(-1)


class GanCmpD(nn.Module):
    def __init__(self, nc, nf):
        super().__init__()
        ng = 2
        self.feature_net = nn.Sequential(
            spectral_init(nn.Conv2d(nc * 2, nf, 4, 2, 1)),
            # GroupNormUnscaled(ng, nf),
            nn.LeakyReLU(0.2, True),
            spectral_init(nn.Conv2d(nf, nf * 2, 4, 2, 1, bias=False)),
            GroupNormUnscaled(ng * 2, nf * 2),
            nn.LeakyReLU(0.2, True),
            AdditiveMultiheadAttention2d(nf * 2, 1, nf * 2 // 8, normalize=False),
            spectral_init(nn.Conv2d(nf * 2, nf * 4, 4, 2, 1, bias=False)),
            GroupNormUnscaled(ng * 4, nf * 4),
            nn.LeakyReLU(0.2, True),
            spectral_init(nn.Conv2d(nf * 4, 1, 4, 1, 0, bias=False)),
        )

    def forward(self, left, right):
        comb = torch.cat([
            torch.cat([left, right], 1),
            torch.cat([right, left], 1),
        ], 0)
        cmp_a, cmp_b = self.feature_net(comb).view(-1).chunk(2, 0)
        return 0.5 * (cmp_a - cmp_b)


class GanCmpDOld(nn.Module):
    def __init__(self, nc, nf):
        super().__init__()
        ng = 2
        self.feature_net = nn.Sequential(
            spectral_init(nn.Conv2d(nc, nf, 4, 2, 1)),
            # GroupNormUnscaled(ng, nf),
            nn.LeakyReLU(0.2, True),
            spectral_init(nn.Conv2d(nf, nf * 2, 4, 2, 1, bias=False)),
            GroupNormUnscaled(ng * 2, nf * 2),
            nn.LeakyReLU(0.2, True),
            spectral_init(nn.Conv2d(nf * 2, nf * 4, 4, 2, 1, bias=False)),
            GroupNormUnscaled(ng * 4, nf * 4),
            nn.LeakyReLU(0.2, True),
            spectral_init(nn.Conv2d(nf * 4, nf * 8, 4, 1, 0, bias=False)),
        )
        cmp_nf = nf * 8
        cmp_ng = ng * 8
        self.cmp_net = nn.Sequential(
            GroupNormUnscaled(cmp_ng * 2, cmp_nf * 2),
            nn.LeakyReLU(0.2, True),
            spectral_init(nn.Linear(cmp_nf * 2, cmp_nf, bias=False)),
            GroupNormUnscaled(cmp_ng, cmp_nf),
            nn.LeakyReLU(0.2, True),
            spectral_init(nn.Linear(cmp_nf, 1, bias=False)),
        )

    def forward(self, left, right):
        comb = torch.cat([left, right], 0)
        features = self.feature_net(comb)
        features_left, features_right = features.view(features.shape[0], -1).chunk(2, 0)
        cmp_input = torch.cat([
            torch.cat([features_left, features_right], -1),
            torch.cat([features_right, features_left], -1)
        ], 0)
        cmp_a, cmp_b = self.cmp_net(cmp_input).view(-1).chunk(2, 0)
        return 0.5 * (cmp_a - cmp_b)


class GanG(nn.Module):
    def __init__(self, nc, nf, nz):
        super().__init__()
        ng = 2
        self.net = nn.Sequential(
            spectral_init(nn.ConvTranspose2d(nz, nf * 4, 4, 1, 0)),
            # GroupNormUnscaled(ng * 4, nf * 4),
            nn.ReLU(True),
            spectral_init(nn.ConvTranspose2d(nf * 4, nf * 2, 4, 2, 1, bias=False)),
            GroupNormUnscaled(ng * 2, nf * 2),
            nn.ReLU(True),
            AdditiveMultiheadAttention2d(nf * 2, 1, nf * 2 // 8, normalize=False),
            spectral_init(nn.ConvTranspose2d(nf * 2, nf, 4, 2, 1, bias=False)),
            GroupNormUnscaled(ng, nf),
            nn.ReLU(True),
            nn.ConvTranspose2d(nf, nc, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, noise):
        noise = noise.unsqueeze(-1).unsqueeze(-1)
        return self.net(noise)


class GradRefiner(nn.Module):
    def __init__(self, nc, nf):
        super().__init__()
        ng = 2
        self.net = nn.Sequential(
            spectral_init(nn.Conv2d(nc * 2, nf, 4, 2, 1)),
            nn.ReLU(True),
            spectral_init(nn.Conv2d(nf, nf * 2, 4, 2, 1, bias=False)),
            GroupNormUnscaled(ng * 2, nf * 2),
            nn.ReLU(True),
            spectral_init(nn.Conv2d(nf * 2, nf * 4, 4, 2, 1, bias=False)),
            GroupNormUnscaled(ng * 4, nf * 4),
            nn.ReLU(True),

            spectral_init(nn.Conv2d(nf * 4, nf * 4, 4, 1, 0)),
            # GroupNormUnscaled(2, nf * 4),
            nn.Tanh(),

            spectral_init(nn.ConvTranspose2d(nf * 4, nf * 4, 4, 1, 0, bias=False)),
            GroupNormUnscaled(ng * 4, nf * 4),
            nn.ReLU(True),
            spectral_init(nn.ConvTranspose2d(nf * 4, nf * 2, 4, 2, 1, bias=False)),
            GroupNormUnscaled(ng * 2, nf * 2),
            nn.ReLU(True),
            spectral_init(nn.ConvTranspose2d(nf * 2, nf, 4, 2, 1, bias=False)),
            GroupNormUnscaled(ng, nf),
            nn.ReLU(True),

            spectral_init(nn.ConvTranspose2d(nf, nc, 4, 2, 1)),
        )

    def forward(self, sample, grad):
        # print(self.net(torch.cat([sample, grad], 1)).shape)
        return self.net(torch.cat([sample, grad], 1))