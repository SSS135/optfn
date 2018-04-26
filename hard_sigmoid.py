def hard_sigmoid(x):
    return x.mul(0.25).add_(0.5).clamp(0, 1)
