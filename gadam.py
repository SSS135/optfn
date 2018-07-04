import math
import torch
import numpy as np
from torch.optim.optimizer import Optimizer


class GAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), nesterov=0.0, avg_sq_mode='weight',
                 amsgrad=False, amsgrad_decay=0, weight_decay=0, l1_decay=0, late_weight_decay=True, eps=1e-8):
        """
        :param avg_sq_mode: 'global' or 'tensor' or 'weight' or 'output'
        """
        defaults = dict(lr=lr, betas=betas, nesterov=nesterov, amsgrad=amsgrad, amsgrad_decay=amsgrad_decay,
                        weight_decay=weight_decay, l1_decay=l1_decay, late_weight_decay=late_weight_decay, eps=eps)
        self.avg_sq_mode = avg_sq_mode
        super().__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        if self.avg_sq_mode == 'global':
            exp_avg_sq_list = []

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]
                amsgrad = group['amsgrad']
                amsgrad_decay = group['amsgrad_decay']

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['prev_shift'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                if amsgrad:
                    torch.max(max_exp_avg_sq * (1 - amsgrad_decay), exp_avg_sq, out=max_exp_avg_sq)
                    if self.avg_sq_mode == 'global':
                        exp_avg_sq_list.append(max_exp_avg_sq.mean())
                else:
                    if self.avg_sq_mode == 'global':
                        exp_avg_sq_list.append(exp_avg_sq.mean())

        if self.avg_sq_mode == 'global':
            global_exp_avg_sq = np.mean(exp_avg_sq_list)

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                amsgrad = group['amsgrad']

                exp_avg = state['exp_avg']

                if self.avg_sq_mode == 'weight':
                    exp_avg_sq = state['max_exp_avg_sq'] if amsgrad else state['exp_avg_sq']
                elif self.avg_sq_mode == 'tensor':
                    exp_avg_sq = (state['max_exp_avg_sq'] if amsgrad else state['exp_avg_sq']).mean()
                elif self.avg_sq_mode == 'output':
                    exp_avg_sq = (state['max_exp_avg_sq'] if amsgrad else state['exp_avg_sq'])
                    exp_avg_sq = exp_avg_sq.view(exp_avg_sq.shape[0], -1).mean(-1)\
                        .view(exp_avg_sq.shape[0], *((exp_avg_sq.dim() - 1) * [1]))
                elif self.avg_sq_mode == 'global':
                    exp_avg_sq = global_exp_avg_sq
                else:
                    raise ValueError()

                beta1, beta2 = group['betas']

                state['step'] += 1

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Decay the first and second moment running average coefficient
                exp_avg = exp_avg / bias_correction1
                exp_avg_sq = exp_avg_sq / bias_correction2

                if self.avg_sq_mode == 'weight' or self.avg_sq_mode == 'output':
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = math.sqrt(exp_avg_sq) + group['eps']

                late_weight_decay = group['late_weight_decay']
                if late_weight_decay:
                    exp_avg = exp_avg.div(denom)

                weight_decay = group['weight_decay']
                l1_decay = group['l1_decay']
                if weight_decay != 0:
                    exp_avg.add_(weight_decay, p.data)
                if l1_decay != 0:
                    exp_avg.add_(l1_decay, p.data.sign())

                if not late_weight_decay:
                    exp_avg = exp_avg.div(denom)

                lr = group['lr']
                nesterov = group['nesterov']
                if nesterov != 0:
                    prev_shift = state['prev_shift']
                    p.data.sub_(nesterov, prev_shift)
                    cur_shift = (-lr / (1 - nesterov)) * exp_avg
                    prev_shift.copy_(cur_shift)
                    p.data.add_(cur_shift)
                else:
                    grad = exp_avg
                    p.data.add_(-lr, grad)

        return loss
