from torch.optim.optimizer import Optimizer
import math


class _OptParamScheduler:
    def __init__(self, optimizer, last_epoch=-1, param_name='lr'):
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        self.param_name = param_name
        self.initial_param_name = 'initial_' + param_name
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault(self.initial_param_name, group[param_name])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if self.initial_param_name not in group:
                    raise KeyError("param '{}' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(self.initial_param_name, i))
        self.base_values = list(map(lambda group: group[self.initial_param_name], optimizer.param_groups))
        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def get_value(self):
        raise NotImplementedError

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        for param_group, value in zip(self.optimizer.param_groups, self.get_value()):
            param_group[self.param_name] = value


class CosineAnnealingParam(_OptParamScheduler):
    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1, param_name='lr'):
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch, param_name)

    def get_value(self):
        return [self.eta_min + (base_value - self.eta_min) *
                (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
                for base_value in self.base_values]


class CosineAnnealingRestartParam(CosineAnnealingParam):
    def __init__(self, optimizer, T_max, T_mult, eta_min=0, last_epoch=-1, param_name='lr'):
        self.T_mult = T_mult
        super().__init__(optimizer, T_max, eta_min, last_epoch, param_name)

    def get_value(self):
        if self.last_epoch >= self.T_max:
            self.last_epoch = 0
            self.T_max = self.T_max * self.T_mult
        return super().get_value()
