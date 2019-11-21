import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn.functional as F


def get_vat_loss(net, inputs, loss_fn, outputs=None, eps=0.1, xi=1e-5):
    r"""
    Args:
        outputs: Output of net(inputs). Computed if None.
    """

    with torch.no_grad():
        outputs = net(inputs).detach() if outputs is None else outputs.detach()

    inputs_adv = get_vat_inputs(net, inputs, loss_fn, outputs, eps, xi)

    with torch.enable_grad():
        outputs_adv = net(inputs_adv)
        err_adv = loss_fn(outputs_adv, outputs)

    return err_adv


def get_vat_inputs(net, inputs, loss_fn, outputs=None, eps=0.1, xi=1e-5):
    r"""
    Args:
        outputs: Output of net(inputs). Computed if None.
    """

    with torch.no_grad():
        outputs = net(inputs).detach() if outputs is None else outputs.detach()

    with torch.enable_grad():
        noise = xi * torch.randn_like(inputs)
        noise.requires_grad = True
        inputs_noisy = inputs + noise

        outputs_noisy = net(inputs_noisy)
        err_noisy = loss_fn(outputs_noisy, outputs)

    with torch.no_grad():
        grad = autograd.grad(err_noisy, noise, only_inputs=True)[0].detach()
        grad_flat = grad.view(grad.shape[0], -1)
        grad = eps * (grad_flat / grad_flat.abs().max(-1, keepdim=True)[0]).view_as(grad)
        return inputs + grad