from ppo_pytorch.common.barron_loss import barron_loss


def l1_quantile_loss(output, target, tau, reduce=True):
    u = target - output
    loss = (tau - (u.detach() <= 0).float()).mul_(u)
    return loss.mean() if reduce else loss


def huber_quantile_loss(output, target, tau, k=0.02, reduce=True):
    u = target - output
    loss = (tau - (u.detach() <= 0).float()).mul_(u.detach().abs().clamp(max=k).div_(k)).mul_(u)
    return loss.mean() if reduce else loss


def barron_quantile_loss(output, target, tau, alpha, c, reduce=True):
    u = target - output
    loss = (tau - (u.detach() <= 0).float()).abs() * barron_loss(output, target, alpha, c, False)
    return loss.mean() if reduce else loss