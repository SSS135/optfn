import torch
from torch.nn import Parameter, Module
from torch.autograd import Variable


def near_instance_norm(input, running_mean, running_std, weight=None, bias=None,
                       training=False, momentum=0.01, near_momentum=0.2, eps=1e-5):
    assert (weight is None) == (bias is None)

    b, c, _, _ = input.shape

    input = input.contiguous()
    input_2d = input.view(b * c, -1)

    inst_mean = input_2d.mean(-1)
    inst_std = input_2d.var(-1).add_(eps).sqrt_()

    if training:
        running_mean.lerp_(inst_mean.data.view(b, c).mean(0), momentum)
        running_std.lerp_(inst_std.data.view(b, c).mean(0), momentum)

    # make mean / var gradients scale independent of near_momentum (though somewhat incorrect)
    inst_mean.data.mul_(near_momentum)
    inst_std.data.mul_(near_momentum)

    near_mean = inst_mean.add_(1 - near_momentum, Variable(running_mean.repeat(b)))
    near_std = inst_std.add_(1 - near_momentum, Variable(running_std.repeat(b)))

    output = input_2d.sub(near_mean.unsqueeze(1)).div_(near_std.unsqueeze(1))

    if weight is not None:
        output.mul_(weight.repeat(b).unsqueeze(1)).add_(bias.repeat(b).unsqueeze(1))

    return output.view_as(input)


class NearInstanceNorm2d(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.01, affine=True, near_momentum=0.3):
        super().__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        self.momentum = momentum
        self.near_momentum = near_momentum
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features))
            self.bias = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        if self.affine:
            self.weight.data.uniform_(0.95, 1.05)
            self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.size(1) != self.running_mean.nelement():
            raise ValueError('got {}-feature tensor, expected {}'
                             .format(input.size(1), self.num_features))

    def forward(self, input):
        self._check_input_dim(input)
        return near_instance_norm(input, self.running_mean, self.running_var, self.weight, self.bias,
                                  self.training, self.momentum, self.near_momentum, self.eps)

    def __repr__(self):
        return ('{name}({num_features}, eps={eps}, momentum={momentum}, affine={affine})'
                .format(name=self.__class__.__name__, **self.__dict__))
