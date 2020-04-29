import torch


def drop_small_grad(parameters, drop_frac=0.9):
    all_grad = torch.cat([p.grad.view(-1) for p in parameters if p.grad is not None])

    rand_idx = torch.randint(high=len(all_grad), size=(32 * 1024,), device=all_grad.device, dtype=torch.int64)
    absgrad = all_grad.abs()[rand_idx]
    sorted = absgrad.sort()[0]
    threshold = sorted[int(len(sorted) * drop_frac)].item()

    for p in parameters:
        if p.grad is None:
            continue
        p.grad[absgrad < threshold] = 0