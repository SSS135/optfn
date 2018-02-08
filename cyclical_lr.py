from torch.optim.lr_scheduler import _LRScheduler
import math


class CyclicalLR(_LRScheduler):
    def __init__(self, optimizer, cycle_len=500, max_lr_mult=5, last_epoch=-1):
        self.cycle_len = cycle_len
        self.max_lr_mult = max_lr_mult
        super(CyclicalLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return map(self._get_lr_single, self.base_lrs)

    def _get_lr_single(self, base_lr):
        cycle = math.floor(1 + self.last_epoch / (2 * self.cycle_len))
        x = abs(self.last_epoch / self.cycle_len - 2 * cycle + 1)
        lr = base_lr * (1 + (self.max_lr_mult - 1) * x)
        return lr
