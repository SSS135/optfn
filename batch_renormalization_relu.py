import torch
from torch.autograd import Variable, Function
from .batch_renormalization import BatchReNorm2d


class BatchReNorm2dReLU(BatchReNorm2d):
    def forward(self, input):
        self._check_input_dim(input)
        return batch_renorm2d_relu(input, self.running_mean, self.running_std, self.weight, self.bias,
                                   self.training, self.momentum, self.eps, self.rmax, self.dmax)


def batch_renorm2d_relu(input, running_mean, running_std, weight, bias,
                        training=True, momentum=0.01, eps=1e-5, rmax=3.0, dmax=5.0):
    return BatchReNorm2dReLUFunction.apply(
        input, running_mean, running_std, weight, bias, training, momentum, eps, rmax, dmax)


class BatchReNorm2dReLUFunction(Function):
    # Note that both forward and backward are @staticmethods
    @staticmethod
    def forward(ctx, input, running_mean, running_std, weight, bias,
                training, momentum, eps, rmax, dmax):
        if not training:
            momentum = 0

        # (C, B * H * W)
        input_1d = input.transpose(0, 1).contiguous().view(input.shape[1], -1)
        sample_mean = input_1d.mean(1)
        sample_std = (input_1d.var(1) + eps).sqrt()

        r = torch.clamp(sample_std / running_std, 1. / rmax, rmax)
        d = torch.clamp((sample_mean - running_mean) / running_std, -dmax, dmax)

        input_normalized = (input - sample_mean.view(1, -1, 1, 1)) / sample_std.view(1, -1, 1, 1)
        input_normalized = input_normalized * r.view(1, -1, 1, 1) + d.view(1, -1, 1, 1)

        running_mean += momentum * (sample_mean - running_mean)
        running_std += momentum * (sample_std - running_std)
        output = input_normalized * weight.view(1, -1, 1, 1) + bias.view(1, -1, 1, 1)

        # relu
        output.clamp(min=0)

        ctx.save_for_backward(input, weight, bias)
        ctx.sample_mean = sample_mean
        ctx.sample_std = sample_std
        ctx.r = r
        ctx.d = d
        ctx.training = training

        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, O_grad):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        assert ctx.training
        input, weight, bias = ctx.saved_variables
        batchMean = Variable(ctx.sample_mean)
        batchStd = Variable(ctx.sample_std)
        r = Variable(ctx.r)
        d = Variable(ctx.d)

        batchMean_u, batchStd_u = batchMean.view(1, -1, 1, 1), batchStd.view(1, -1, 1, 1)
        r_u, d_u = r.view(1, -1, 1, 1), d.view(1, -1, 1, 1)
        weight_u, bias_u = weight.view(1, -1, 1, 1), bias.view(1, -1, 1, 1)

        # xHat(n, c, h, w) = (I(n, c, h, w) - batchMean(c)) / batchStd(c) * r(c) + d(c)
        input_centered = input - batchMean_u
        xHat = input_centered.div(batchStd_u).mul_(r_u).add_(d_u)
        output_before_relu = (xHat * weight_u).add_(bias_u)
        # relu
        O_grad = O_grad.clone()
        O_grad[output_before_relu.data < 0] = 0
        # xHat_grad(nn, c, hh, ww) = O_grad(nn, c, hh, ww) * weight(c)
        xHat_grad = O_grad * weight_u
        # batchStd_grad(c) +=! xHat_grad(nn, c, hh, ww) * (I(nn, c, hh, ww) - batchMean(c))
        batchStd_grad = input_centered.mul(xHat_grad)
        batchStd_grad = batchStd_grad.sum(0).view(batchStd_grad.shape[1], -1).sum(-1)
        # batchStd_grad(c) = batchStd_grad(c) * -r(c) / (batchStd(c) * batchStd(c))
        batchStd_grad.mul_(-r).div_(batchStd).div_(batchStd)
        batchStd_grad_u = batchStd_grad.view(1, -1, 1, 1)
        # batchMean_grad(c) +=! xHat_grad(nn, c, hh, ww)
        batchMean_grad = xHat_grad.sum(0).view(xHat_grad.shape[1], -1).sum(-1)
        # batchMean_grad(c) = batchMean_grad(c) * -r(c) / batchStd(c)
        batchMean_grad.mul_(-r).div_(batchStd)
        batchMean_grad_u = batchMean_grad.view(1, -1, 1, 1)
        # weight_grad(c) +=! O_grad(nn, c, hh, ww) * xHat(nn, c, hh, ww)
        weight_grad = xHat.mul_(O_grad).sum(0).view(xHat.shape[1], -1).sum(-1)
        # bias_grad(c) +=! O_grad(nn, c, hh, ww)
        bias_grad = O_grad.sum(0).view(O_grad.shape[1], -1).sum(-1)
        # I_grad(n, c, h, w) = xHat_grad(n, c, h, w) * r(c) / batchStd(c)
        #   + batchStd_grad(c) * (I(n, c, h, w) - batchMean(c)) / (batchStd(c) * N * H * W)
        #   + batchMean_grad(c) * (1 / (N * H * W))
        NHW = input.shape[0] * input.shape[2] * input.shape[3]
        I_grad = xHat_grad.mul_(r_u).div_(batchStd_u)
        I_grad.add_(input_centered.mul_(batchStd_grad_u).div_(batchStd_u).mul_(1 / NHW))
        I_grad.add_(batchMean_grad_u.mul(1 / NHW))

        return I_grad, None, None, weight_grad, bias_grad, None, None, None, None, None