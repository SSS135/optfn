import torch
import torch.autograd as autograd
from torch.autograd import Variable
from .kl_div import kl_div


def get_vat_loss(net, inputs, outputs=None, eps=0.1, xi=1e-6,
                 swap_1=False, swap_2=False, ent_1=False, ent_2=True, dual_1=True, dual_2=True):
    r"""
    Args:
        outputs: Output of net(inputs). Computed if None.
    """

    bs = inputs.size(0)
    in_src_size = inputs.size()
    outputs = net(inputs) if outputs is None else outputs

    inputs = inputs.view(bs, -1)
    outputs = outputs.view(bs, -1)

    noise = xi * torch.randn(inputs.size()).type_as(inputs.data)
    noise = autograd.Variable(noise, requires_grad=True)
    inputs_noisy = Variable(inputs.data) + noise

    outputs_noisy = net(inputs_noisy.view(*in_src_size)).view(bs, -1)
    err_noisy = kl_div(Variable(outputs.data), outputs_noisy, ent_1, dual_1, swap_1)

    grad = autograd.grad(err_noisy, noise)[0]
    norm = ((grad**2).sum(1) / (grad.size(1) - 1))**0.5
    grad = eps * grad / norm.unsqueeze(1)

    inputs_adv = inputs + Variable(grad.data)
    outputs_adv = net(inputs_adv.view(*in_src_size)).view(bs, -1)
    err_adv = kl_div(Variable(outputs.data), outputs_adv, ent_2, dual_2, swap_2)

    return err_adv