import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn.functional as F


def get_vat_loss(net, inputs, loss_fn, eps=0.1):
    inputs_adv = get_vat_inputs(net, inputs, loss_fn, eps)

    with torch.enable_grad():
        outputs_adv = net(inputs_adv)
        err_adv = loss_fn(outputs_adv)

    return err_adv


def get_vat_inputs(net, inputs, loss_fn, eps=0.1):
    inputs_g = inputs.data.detach()
    inputs_g.requires_grad = True

    with torch.enable_grad():
        outputs_g = net(inputs_g)
        err_noisy = loss_fn(outputs_g)

    with torch.no_grad():
        grad = autograd.grad(err_noisy, inputs_g, only_inputs=True)[0].detach()
        grad_flat = grad.view(grad.shape[0], -1)
        grad = eps * (grad_flat / grad_flat.abs().max(-1, keepdim=True)[0]).view_as(grad)
        return inputs + grad