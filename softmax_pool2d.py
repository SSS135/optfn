import numpy as np
import torch.nn.functional as F


def softmax_pool2d(x, size, subtract_min=True):
    pad = np.asarray(x.shape[-2:])
    pad = pad % size
    x = F.pad(x, (int(pad[0]), 0, int(pad[1]), 0), mode='reflect')
    out_size = np.asarray(x.shape[-2:]) // size
    x = x.view(x.shape[0], x.shape[1], int(out_size[0]), int(out_size[1]), -1)
    x = x * F.softmax(x - x.min() if subtract_min else x, -1)
    x = x.sum(-1)
    return x