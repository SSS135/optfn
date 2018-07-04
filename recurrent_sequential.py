import torch
import torch.nn as nn
import torch.nn.functional as F


class RecurrentSequential(nn.Sequential):
    def __init__(self, *args, unsqueeze_input=False, join_output_and_memory=False):
        super().__init__(*args)
        self.unsqueeze_input = unsqueeze_input
        self.join_output_and_memory = join_output_and_memory

    def forward(self, x, memory):
        mem_idx = 0
        new_memory = []
        for module in self._modules.values():
            if isinstance(module, RecurrentMarker):
                if self.unsqueeze_input:
                    x = x.unsqueeze(0)
                x, new_m = module(x, memory[mem_idx] if memory is not None else None)
                if self.unsqueeze_input:
                    x = x.squeeze(0)
                if self.join_output_and_memory:
                    new_m = (x, new_m)
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