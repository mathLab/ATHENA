import autograd.numpy as np
from autograd import elementwise_grad as egrad
from scipy import linalg
import matplotlib.pyplot as plt
from functools import partial

from athena.active import ActiveSubspaces
from athena.utils import Normalizer

# custom 2d outputs of interest
from numpy_functions import cubic_2d, sin_2d, exp_2d

import warnings
warnings.filterwarnings('ignore')


np.random.seed(42)

# Global parameters
n_samples = 400
input_dim = 2

# Uniformly distributed inputs
lb = 0 * np.ones(input_dim) # lower bounds
ub = 2 * np.ones(input_dim) # upper bounds

def inputs_uniform(n_samples, lb,  ub):
    return np.vstack(
        np.array([np.random.uniform(lb[i], ub[i], n_samples) for i in range(lb.shape[0])]).T)

# Gaussian model for the inputs
mean = np.ones(input_dim)
cov = 0.5*np.diag(np.ones(input_dim))

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
# x_raw = inputs_uniform(n_samples, lb, ub)
# nor = Normalizer(lb, ub)
# x = nor.fit_transform(x_raw)
x_raw = inputs_gaussian(n_samples, mean, cov)
x = (x_raw-mean).dot(linalg.sqrtm(np.linalg.inv(cov)))

# Define the output of interest and compute the gradients
# func = partial(output, normalizer=nor, r=generatrix)
func = sin_2d
f = func(x)
df = egrad(func)(x)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x[:,0], x[:,1], f)
plt.show()

# compute the active subspace
asub = ActiveSubspaces(dim=1, method='exact', n_boot=100)
asub.fit(gradients=df)

title = '2D sine'
asub.plot_eigenvalues(figsize=(6, 4), title=title)
print("Eigenvalues: {}".format(np.squeeze(asub.evals)))

asub.plot_eigenvectors(figsize=(6, 4), title=title)
asub.plot_sufficient_summary(x, f, figsize=(6, 4), title=title)

asub_2d = ActiveSubspaces(dim=2, method='exact', n_boot=100)
asub_2d.fit(gradients=df)
asub_2d.plot_sufficient_summary(x, f, figsize=(6, 4), title=title)
