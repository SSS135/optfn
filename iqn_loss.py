
def l1_quantile_loss(output, target, tau, reduce=True):
    u = target - output
    loss = (tau - (u.detach() <= 0).float()).mul_(u)
    return loss.mean() if reduce else loss


def huber_quantile_loss(output, target, tau, k=0.02, reduce=True):
    u = target - output
    loss = (tau - (u.detach() <= 0).float()).mul_(u.detach().abs().clamp(max=k).div_(k)).mul_(u)
    return loss.mean() if reduce else loss