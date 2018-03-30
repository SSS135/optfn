import math
import torch
import numpy as np
from torch.optim.optimizer import Optimizer


class GAdam(Optimizer):
    def __init__(self, params, lr=0.01, betas=(0.9, 0.999), nesterov=0.0,
                 avg_sq_mode='weight', amsgrad=False, weight_decay=0, eps=1e-6):
        """
        :param avg_sq_mode: 'global' or 'tensor' or 'weight' or 'output'
        """
        defaults = dict(lr=lr, betas=betas, nesterov=nesterov, amsgrad=amsgrad, eps=eps, weight_decay=weight_decay)
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

        # if self.avg_sq_mode == 'global':
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
                exp_avg_sq.mul_(beta2).add_(1 - beta2, grad.pow(2).add_(group['eps']).log_())

                assert not amsgrad
                if amsgrad:
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # if self.avg_sq_mode == 'global':
                    exp_avg_sq_list.append(max_exp_avg_sq.mean())
                else:
                    # if self.avg_sq_mode == 'global':
                    exp_avg_sq_list.append(exp_avg_sq.mean())

        # if self.avg_sq_mode == 'global':
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
                # elif self.avg_sq_mode == 'global':
                #     exp_avg_sq = global_exp_avg_sq
                else:
                    raise ValueError()

                beta1, beta2 = group['betas']

                state['step'] += 1

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Decay the first and second moment running average coefficient
                exp_avg = exp_avg / bias_correction1
                denom = exp_avg_sq.mul(-0.25 / bias_correction2).exp_() #bias_correction2

                lr = group['lr']
                nesterov = group['nesterov']
                prev_delta = state['prev_delta']

                # if self.avg_sq_mode == 'weight' or self.avg_sq_mode == 'output':
                #     denom = exp_avg_sq.sqrt().sqrt()
                # else:
                #     denom = math.sqrt(math.sqrt(exp_avg_sq))

                exp_avg = exp_avg.mul(denom)

                weight_decay = group['weight_decay']
                if weight_decay != 0:
                    exp_avg.add_(weight_decay, p.data)

                if nesterov != 0:
                    grad = exp_avg.add(-nesterov, prev_delta)
                    prev_delta.copy_(exp_avg)
                else:
                    grad = exp_avg

                p.data.add_(-lr / (1 - nesterov), grad)

        return loss
