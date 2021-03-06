import torch.nn as nn

def swish(x):
    return x * x.sigmoid()


class Swish(nn.Module):
    def forward(self, x):
        return swish(x)
