import torch
import math

def radial(x, generatrix, normalizer=None):
    if normalizer:
        x = normalizer.inverse_transform(x)
    return generatrix(torch.norm(x, dim=1)**2)

def sin_2d(x, normalizer=None):
    if normalizer:
        x = normalizer.inverse_transform(x)
    return 0.5*torch.sin(2*math.pi*(torch.sum(x, dim=1)))+1

def exp_2d(x, normalizer=None):
    if normalizer:
        x = normalizer.inverse_transform(x)
    return torch.exp(-(x[:, 0]-0.5)**2-x[:, 1]**2)

def cubic_2d(x, normalizer=None):
    if normalizer:
        x = normalizer.inverse_transform(x)
    return x[:, 0]**3+x[:, 1]**3+0.2*x[:, 0]+0.6*x[:, 1]