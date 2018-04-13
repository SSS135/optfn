import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .conv_chunk import ConvChunk2d


class CapsuleLayer(nn.Module):
    def __init__(self, in_channels, num_capsules, caps_channels, kernel_size, stride, padding, num_iterations=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.num_capsules = num_capsules
        self.caps_channels = caps_channels
        self.num_iterations = num_iterations
        self.chunk = ConvChunk2d(in_channels, kernel_size, stride, padding)
        self.weight = nn.Parameter(torch.randn(1, 1, 1, kernel_size, kernel_size, in_channels, num_capsules * caps_channels))
        self.bias = nn.Parameter(torch.zeros(1, 1, 1, kernel_size, kernel_size, 1, num_capsules * caps_channels))

    def squash(self, tensor, dim=-1):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / squared_norm.sqrt()

    def forward(self, input):
        # (B, K, K, C, OH, OW)
        chunks = self.chunk(input)
        # (B, OH, OW, K, K, 1, C)
        chunks = chunks.permute(0, 4, 5, 1, 2, 3).unsqueeze(-2)
        # (B, OH, OW, K, K, 1, C) @ (1, 1, 1, K, K, C, NC * CS) = (B, OH, OW, K, K, 1, NC * CS)
        priors = chunks @ self.weight
        priors += self.bias
        # (B, OH, OW, K * K, NC, CS)
        priors = priors.view(*priors.shape[:3], self.kernel_size * self.kernel_size, self.num_capsules, self.caps_channels)
        # # (B, OH, OW, NC * CS)
        # x_hat = x_hat.mean(-3).view(*x_hat.shape[:3], -1)
        # x_hat = x_hat.permute(0, 3, 1, 2)

        # (B, OH, OW, K * K, NC, 1)
        logits = Variable(priors.data.new(*priors.shape[:-1], 1).zero_())
        for i in range(self.num_iterations):
            probs = F.softmax(logits, dim=3)
            # (B, OH, OW, 1, NC, CS)
            outputs = self.squash((probs * priors).sum(dim=3, keepdim=True))

            if i != self.num_iterations - 1:
                # (B, OH, OW, K * K, NC, 1)
                delta_logits = (priors * outputs).sum(dim=-1, keepdim=True)
                logits = logits + delta_logits

        # (B, OH, OW, NC * CS)
        outputs = outputs.view(*outputs.shape[:3], self.num_capsules * self.caps_channels)
        outputs = outputs.permute(0, 3, 1, 2)

        return outputs.contiguous()