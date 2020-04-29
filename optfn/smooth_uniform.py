from optfn.gaussian_blur import GaussianBlur2d
import torch
import torch.nn as nn
import torch.nn.functional as F


class SmoothUniform(nn.Module):
    def __init__(self, size, num_channels, kernel_size, sigma):
        super().__init__()
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.size = size
        self.blur = GaussianBlur2d(num_channels, kernel_size, sigma)

    def forward(self, batches):
        with torch.no_grad():
            nsize = self.size + self.kernel_size - 1
            noise = torch.randn(batches, self.num_channels, nsize, nsize, device=self.blur.weight.device)
            noise = self.blur(noise)
            noise = (noise - noise.mean()) / (noise.std() + 1 / self.kernel_size)
            noise = torch.erf(noise) * 0.5 + 0.5
        return noise