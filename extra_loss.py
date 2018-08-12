import torch.nn as nn
import torch


def get_extra_loss(module: nn.Module, **weights):
    loss = 0
    for m in module.modules():
        if not hasattr(m, 'extra_loss') or m.extra_loss_name not in weights:
            continue
        loss += weights[m.extra_loss_name] * m.extra_loss.mean()
    return loss
