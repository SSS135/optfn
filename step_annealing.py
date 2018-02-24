from torch.optim.lr_scheduler import MultiStepLR


class StepAnnealingLR(MultiStepLR):
    def __init__(self, optimizer, T_max, T_mult, milestones, gamma=0.1, last_epoch=-1):
        self.T_max = T_max
        self.T_mult = T_mult
        super().__init__(optimizer, milestones, gamma, last_epoch)

    def get_lr(self):
        if self.last_epoch >= self.T_max:
            self.last_epoch = 0
            self.T_max = self.T_max * self.T_mult
            self.milestones = [m * self.T_mult for m in self.milestones]
        return super().get_lr()
