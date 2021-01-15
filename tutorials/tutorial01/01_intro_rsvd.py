import autograd.numpy as np
from autograd import elementwise_grad as egrad
from scipy import linalg
import matplotlib.pyplot as plt
from functools import partial
import time
from athena.active import ActiveSubspaces
from athena.utils import Normalizer

# custom 2d outputs of interest
from numpy_functions import cubic_2d, sin_2d, exp_2d

import warnings
warnings.filterwarnings('ignore')


np.random.seed(42)

# Global parameters
n_samples = 100
input_dim = 100000

# Uniformly distributed inputs
lb = 0 * np.ones(input_dim) # lower bounds
ub = 2 * np.ones(input_dim) # upper bounds

def inputs_uniform(n_samples, lb,  ub):
    return np.vstack(
        np.array([np.random.uniform(lb[i], ub[i], n_samples) for i in range(lb.shape[0])]).T)

def inputs_gaussian(n_samples, mean, cov):
    return np.random.multivariate_normal(mean, cov, n_samples)


# Radial symmetric output of interest
def generatrix(x):
    return np.sin(x)

def output(x, normalizer, r):
    if normalizer:
        return r(np.linalg.norm(normalizer.inverse_transform(x), axis=1)**2)
    else:
        return r(np.linalg.norm(x, axis=1)**2)


# Generate and normalize inputs
x_raw = inputs_uniform(n_samples, lb, ub)
nor = Normalizer(lb, ub)
x = nor.fit_transform(x_raw)
# x_raw = inputs_gaussian(n_samples, mean, cov)
# x = (x_raw-mean).dot(linalg.sqrtm(np.linalg.inv(cov)))

# Define the output of interest and compute the gradients
# func = partial(output, normalizer=nor, r=generatrix)
func = sin_2d
f = func(x)
df = egrad(func)(x)

# compute the active subspace
start = time.time()
asub = ActiveSubspaces(dim=2, method='exact', n_boot=100)
asub.fit(gradients=df)
print("time", time.time()-start)
asub.plot_eigenvalues()
asub.plot_sufficient_summary(x, f)
print(asub.W2)