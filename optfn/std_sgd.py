import torch
from torch.optim.optimizer import Optimizer, required
import torch.nn as nn


class SSGD(Optimizer):
    def __init__(self, module, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=0.5):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        super(SSGD, self).__init__(module.parameters(), defaults)
        self.param_to_std_map = {}
        for m in module.modules():
            m.register_backward_hook

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        prev_d_p = param_state['prev_d_p'] = torch.zeros_like(p.data)
                        buf = buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        prev_d_p = param_state['prev_d_p']
                        buf = buf.mul_(momentum).add_(1 - dampening, d_p)

                    d_p = buf - prev_d_p.mul_(nesterov)
                    prev_d_p.copy_(buf)

                p.data.add_(-group['lr'], d_p)

        return loss
