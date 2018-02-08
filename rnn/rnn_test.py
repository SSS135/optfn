import numpy as np
import numpy.random as rng
import torch
import torch.nn.utils
import torch.utils.data
from torch.autograd import Variable


def get_sin_sum(x, count, min_period=0.5, max_period=2, min_scale=0.5, max_scale=2):
    x = np.atleast_1d(x).reshape(-1)

    offsets = rng.rand(count) * np.pi * 2
    periods = min_period + (max_period - min_period) * rng.rand(count)
    scales = min_scale + (max_scale - min_scale) * rng.rand(count)

    periods = np.tile(periods, (len(x), 1))
    offsets = np.tile(offsets, (len(x), 1))
    scales = np.tile(scales, (len(x), 1))

    return (np.sin(x * periods.T + offsets.T) * scales.T).sum(0)


def get_wsum(seq_count, seq_len):
    w = rng.rand(seq_len) * 2 - 1
    x = rng.randn(seq_count, seq_len)
    y = np.mean(x * w, 1)
    return x, y


def get_reverse(seq_count, seq_len):
    x = rng.rand(seq_count, seq_len)*2 - 1
    z = np.zeros((seq_count, seq_len))
    y = np.flip(x, 1)
    x = np.append(x, z, 1)
    return x, y


def get_masked_sum(seq_count, seq_len):
    xv = rng.rand(seq_count, seq_len)
    xm = np.zeros((seq_count, seq_len))
    y = np.empty((seq_count, 1))
    for i in range(seq_count):
        ri = np.random.choice(seq_len, 2, replace=False)
        xm[i, ri] = 1
        y[i] = xv[i, ri].sum()
    x = np.stack((xv, xm), axis=2)
    return x, y


def split_to_sequences(x, y, seq_len, seq_interval):
    assert x.shape[0] == y.shape[0]
    seqs_count = (x.shape[0] - seq_len) // seq_interval
    seqs_x = np.empty((seqs_count, seq_len, x.shape[1]))
    seqs_y = np.empty((seqs_count, seq_len, y.shape[1]))
    for i in range(seqs_count):
        sl = slice(i * seq_interval, i * seq_interval + seq_len)
        seqs_x[i], seqs_y[i] = x[sl], y[sl]
    return seqs_x, seqs_y


def create_dataloader(x, y, batch_size, cuda):
    x = torch.FloatTensor(x)
    y = torch.FloatTensor(y)
    dataset = torch.utils.data.TensorDataset(x, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size, pin_memory=cuda)
    return loader


def fit(model, loss_fn, optimizer, loader, iters, log_interval=None):
    min_loss = None
    for i in range(iters):
        loss = fit_epoch(model, loss_fn, optimizer, loader)
        if min_loss == None or loss < min_loss:
            min_loss = loss
        if log_interval is not None and (i % log_interval == 0 or i + 1 == iters):
            print(i + 1, loss)
    return min_loss


def fit_epoch(model, loss_fn, optimizer, loader):
    cuda = next(model.parameters()).is_cuda
    model._train()
    for batch_idx, (data, target) in enumerate(loader):
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)

        data = data.transpose(1, 0)
        target = target.transpose(1, 0)

        h = model.create_state(data.size(1))
        target_len = target.size(0)

        optimizer.zero_grad()
        out = model(data, h)
        out, h = out if type(out) is tuple else (out, out)
        #print(out.size(), target.size(), target_len)
        loss = loss_fn(out[-target_len - 1: -1], target)
        loss.backward()
        optimizer.step()
    return loss.data.mean()


def fit_epoch_cell(model, loss_fn, optimizer, loader):
    cuda = next(model.parameters()).is_cuda
    model._train()
    for batch_idx, (data, target) in enumerate(loader):
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        h = model.create_state(data.size(0))

        optimizer.zero_grad()
        losses = []
        seq_len = data.size(1)
        for seq_i in range(seq_len):
            seq = data[:, seq_i]
            out = model(seq, h)
            out, h = out if type(out) is tuple else (out, out)
            if seq_i > seq_len/2:
                loss = loss_fn(out, target[:, seq_i])
                losses.append(loss)
        torch.autograd.backward(losses)
        optimizer.step()
    return np.mean([l.data.cpu().numpy() for l in losses])