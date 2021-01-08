import autograd.numpy as np

def radial(x, generatrix, normalizer=None):
    if normalizer:
        x = normalizer.inverse_transform(x)
    return generatrix(np.linalg.norm(x, axis=1)**2)

def sin_2d(x, normalizer=None):
    if normalizer:
        x = normalizer.inverse_transform(x)
    return 0.5*np.sin(2*np.pi*(x[:, 0]+x[:, 1]))+1

def exp_2d(x, normalizer=None):
    if normalizer:
        x = normalizer.inverse_transform(x)
    return np.exp(-(x[:, 0]-0.5)**2-x[:, 1]**2)

def cubic_2d(x, normalizer=None):
    if normalizer:
        x = normalizer.inverse_transform(x)
    return x[:, 0]**3+x[:, 1]**3+0.2*x[:, 0]+0.6*x[:, 1]