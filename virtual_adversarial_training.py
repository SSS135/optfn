import torch
import torch.autograd as autograd
from torch.autograd import Variable
from .kl_div import kl_div


# def get_vat_loss(net, inputs, outputs=None, eps=0.1, xi=1e-6,
#                  swap_1=False, swap_2=False, ent_1=False, ent_2=True, dual_1=True, dual_2=True):
#     r"""
#     Args:
#         outputs: Output of net(inputs). Computed if None.
#     """
#
#     with torch.no_grad():
#         bs = inputs.shape[0]
#         in_src_size = inputs.shape
#         inputs = inputs.detach()
#         outputs = net(inputs) if outputs is None else outputs
#         outputs = outputs.detach()
#
#         inputs = inputs.view(bs, -1)
#         outputs = outputs.view(bs, -1)
#
#         noise = xi * torch.randn_like(inputs)
#         noise.requires_grad = True
#         inputs_noisy = inputs + noise
#
#     with torch.enable_grad():
#         outputs_noisy = net(inputs_noisy.view(*in_src_size)).view(bs, -1)
#         err_noisy = kl_div(outputs, outputs_noisy, ent_1, dual_1, swap_1)
#
#         grad = autograd.grad(err_noisy, noise, only_inputs=True)[0]
#         norm = grad.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt() # ((grad**2).sum(1) / (grad.size(1) - 1))**0.5
#         grad = eps * grad / norm # .unsqueeze(1)
#
#         inputs_adv = inputs + grad.detach()
#         outputs_adv = net(inputs_adv.view(*in_src_size)).view(bs, -1)
#         err_adv = kl_div(outputs, outputs_adv, ent_2, dual_2, swap_2)
#
#     return err_adv


def get_vat_loss(net, inputs, outputs=None, eps=0.1, xi=1e-6, custom_kl=kl_div):
    r"""
    Args:
        outputs: Output of net(inputs). Computed if None.
    """

    with torch.no_grad():
        bs = inputs.shape[0]
        in_src_size = inputs.shape
        inputs = inputs.detach()
        outputs = net(inputs) if outputs is None else outputs
        outputs = outputs.detach()

        inputs = inputs.view(bs, -1)
        outputs = outputs.view(bs, -1)

    with torch.enable_grad():
        noise = xi * torch.randn_like(inputs)
        noise.requires_grad = True
        inputs_noisy = inputs + noise

        outputs_noisy = net(inputs_noisy.view(*in_src_size)).view(bs, -1)
        err_noisy = custom_kl(outputs, outputs_noisy)

        grad = autograd.grad(err_noisy, noise, only_inputs=True)[0]
        norm = grad.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt()  # ((grad**2).sum(1) / (grad.size(1) - 1))**0.5
        grad = eps * grad / norm  # .unsqueeze(1)

        inputs_adv = inputs + grad.detach()
        outputs_adv = net(inputs_adv.view(*in_src_size)).view(bs, -1)
        err_adv = custom_kl(outputs, outputs_adv)

    return err_adv