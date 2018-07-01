import torch
import torch.nn as nn
import torch.nn.functional as F


class RecurrentSequential(nn.Sequential):
    def forward(self, x, memory):
        mem_idx = 0
        new_memory = []
        for module in self._modules.values():
            if isinstance(module, RecurrentMarker):
                x, new_m = module(x, memory[mem_idx] if memory is not None else None)
                mem_idx += 1
                new_memory.append(new_m)
            else:
                x = module(x)
        return x, tuple(new_memory)


class RecurrentMarker(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)