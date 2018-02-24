from torch.optim.lr_scheduler import CosineAnnealingLR as CALR
from torch.optim.lr_scheduler import _LRScheduler
import math
import random


class CosineAnnealingRestartLR(CALR):
    def __init__(self, optimizer, T_max, T_mult, eta_min=0, last_epoch=-1):
        self.T_mult = T_mult
        super().__init__(optimizer, T_max, eta_min, last_epoch)

    def get_lr(self):
        if self.last_epoch >= self.T_max:
            self.last_epoch = 0
            self.T_max = self.T_max * self.T_mult
        return super().get_lr()


class StochasticCosineLR(_LRScheduler):
    def __init__(self, optimizer, eta_min=0, last_epoch=-1):
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        rand = random.random()
        return [self.eta_min + (base_lr - self.eta_min) *
                (1 + math.cos(math.pi * rand)) / 2
                for base_lr in self.base_lrs]