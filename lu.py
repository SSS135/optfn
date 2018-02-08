import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, Function
import numpy as np


class LearnedLU(nn.Module):
    def __init__(self, y, xmin, xmax, smooth_loss=0):
        super(LearnedLU, self).__init__()
        self.xmin = xmin
        self.xmax = xmax
        self.y = nn.Parameter(y)
        self.smooth_loss = smooth_loss

    def forward(self, x):
        if self.y.grad is None:
            (self.y.sum()*0).backward()
        return LU(self.y, self.xmin, self.xmax, self.smooth_loss)(x)

    def to_constant(self):
        return LU(self.y.data, self.xmin, self.xmax)


class LU(Function):
    def __init__(self, y, xmin, xmax, smooth_loss=0):
        super(LU, self).__init__()
        assert torch.is_tensor(y) or isinstance(y, Variable) and y.requires_grad
        self.y_data = y if torch.is_tensor(y) else y.data
        self.y_param = None if torch.is_tensor(y) else y
        self.xmin, self.xmax = xmin, xmax
        self.smooth_loss = smooth_loss

    def forward(self, x):
        self.save_for_backward(x)
        x = x.contiguous()
        return self._sample_lerp(x)

    def backward(self, grad_y):
        x, = self.saved_tensors
        x = x.contiguous()
        grad_x = grad_y*self._derivative(x)
        if self.y_param is not None:
            self._fill_grad(x, grad_y)
        return grad_x

    def to_learned(self):
        return LearnedLU(self.y_data, self.xmin, self.xmax)

    @staticmethod
    def replicate(func, min, max, steps, cuda, dtype=torch.FloatTensor):
        x = torch.linspace(min, max, steps).type(dtype)
        if cuda:
            x = x.cuda()
        y = func(Variable(x)).data
        return LU(y, min, max)

    def get_factory(self):
        return lambda x: LU(self.y_data, self.xmin, self.xmax)(x)

    @staticmethod
    def limit_y(y, xmin, xmax, ymin, ymax):
        size_add = (xmax - xmin)/y.size(0)*2
        if ymin is not None:
            xmin -= size_add
            y = torch.cat((y.new([ymin, ymin]), y.clamp(min=ymin)))
        if ymax is not None:
            xmax += size_add
            y = torch.cat((y.clamp(max=ymax), y.new([ymax, ymax])))
        return y

    def _sample_lerp(self, x):
        x, x_size = x.view(-1), x.size()
        x = (x - self.xmin) / (self.xmax - self.xmin) * (self.y_data.size(0) - 1)
        min_index = x.floor().long().clamp(0, self.y_data.size(0) - 2)
        min_y = self.y_data[min_index]
        max_y = self.y_data[min_index + 1]
        frac = x - min_index.type_as(x)
        res = min_y + (max_y - min_y)*frac
        return res.view(x_size)

    def _derivative(self, x):
        x, x_size = x.view(-1), x.size()
        x = (x - self.xmin) / (self.xmax - self.xmin) * (self.y_data.size(0) - 1)
        min_index = x.floor().long().clamp(0, self.y_data.size(0) - 2)
        min_y = self.y_data[min_index]
        max_y = self.y_data[min_index + 1]
        grad = (max_y - min_y)*(self.y_data.size(0) - 1) / (self.xmax - self.xmin)
        return grad.view(x_size)

    def _fill_grad(self, x, grad):
        x = x.view(-1)
        grad = grad.contiguous().view(-1)
        x = (x - self.xmin) / (self.xmax - self.xmin) * (self.y_param.size(0) - 1)
        min_index = x.floor().long().clamp(0, self.y_param.size(0) - 2)
        frac = (x - min_index.type_as(x)).clamp(0, 1)
        pgrad = self.y_param.grad.data

        for i in range(self.y_param.size(0) - 1):
            mask = min_index == i
            gradm = grad[mask]
            fracm = frac[mask]
            pgrad[i] += ((1 - fracm) * gradm).sum()
            pgrad[i + 1] += (fracm * gradm).sum()

        pgrad[1:-1] += self.smooth_loss / self.y_data.size(0) * self._get_diff(self.y_data)

        # pgrad -= self.y_data/(self.y_data/pgrad).mean()
        # print((self.y_data*pgrad).mean())

    def _get_diff(self, w):
        w_diff = w[:-1] - w[1:]
        w_diff = w_diff[1:] - w_diff[:-1]
        return w_diff