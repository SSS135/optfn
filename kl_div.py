import torch


def kl_div(log_p, log_q, remove_entropy=True, dual=False, swap_args=False):
    if swap_args:
        log_p, log_q = (log_q, log_p)
    if dual:
        return 0.5 * kl_div(log_p, log_q, remove_entropy, False) + \
               0.5 * kl_div(log_q, log_p, remove_entropy, False)

    p = torch.exp(log_p)
    if remove_entropy:
        return (p * (log_p - log_q)).sum(1).mean()
    else:
        return -(p * log_q).sum(1).mean()