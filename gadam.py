import math
import torch
import numpy as np
from torch.optim.optimizer import Optimizer


class GAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), nesterov=0.0, norm_weight_decay=False,
                 avg_sq_mode='weight', amsgrad=False, weight_decay=0, eps=1e-8):
        """
        :param avg_sq_mode: 'global' or 'tensor' or 'weight'
        """
        defaults = dict(lr=lr, betas=betas, nesterov=nesterov, amsgrad=amsgrad, eps=eps,
                        weight_decay=weight_decay, norm_weight_decay=norm_weight_decay)
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

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['prev_delta'] = torch.zeros_like(p.data)
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
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
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

                lr = group['lr']
                nesterov = group['nesterov']
                prev_delta = state['prev_delta']

                if self.avg_sq_mode == 'weight':
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = math.sqrt(exp_avg_sq) + group['eps']

                exp_avg = exp_avg.div(denom)

                weight_decay = group['weight_decay']
                if weight_decay != 0:
                    if group['norm_weight_decay']:
                        decay = weight_decay * p.data
                        decay -= decay.sign() * decay.abs().mean()
                        decay *= math.sqrt(p.data.pow(2).mean())
                        exp_avg += decay
                    else:
                        exp_avg.add_(weight_decay, p.data)

                if nesterov != 0:
                    grad = exp_avg.add(-nesterov, prev_delta)
                    prev_delta.copy_(exp_avg)
                else:
                    grad = exp_avg

                p.data.add_(-lr / (1 - nesterov), grad)

        return loss
