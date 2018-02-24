import math
import torch
from torch.optim.optimizer import Optimizer


class NAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), nesterov=0.5, amsgrad=False, weight_decay=0, eps=1e-8):
        defaults = dict(lr=lr, betas=betas, nesterov=nesterov, amsgrad=amsgrad, eps=eps, weight_decay=weight_decay)
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

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['prev_exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Decay the first and second moment running average coefficient
                exp_avg = exp_avg.mul_(beta1).add_(1 - beta1, grad) / bias_correction1
                exp_avg_sq = exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.div(bias_correction2).sqrt_().add_(group['eps'])
                else:
                    denom = exp_avg_sq.div(bias_correction2).sqrt_().add_(group['eps'])

                lr = group['lr']
                nesterov = group['nesterov']
                prev_exp_avg = state['prev_exp_avg']

                exp_avg = exp_avg.div(denom)

                if group['weight_decay'] != 0:
                    exp_avg.add_(group['weight_decay'], p.data)

                delta = exp_avg.add(-nesterov, prev_exp_avg)
                prev_exp_avg.copy_(exp_avg)

                p.data.add_(-lr / (1 - nesterov), delta)

        return loss
