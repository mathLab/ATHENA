
import autograd.numpy as np
from autograd import elementwise_grad as egrad
import matplotlib.pyplot as plt
from functools import partial
import GPy
from athena.active import ActiveSubspaces
from athena.utils import Normalizer, local_linear_gradients
from numpy_functions import cubic_2d, sin_2d, exp_2d, radial

import warnings
warnings.filterwarnings('ignore')


np.random.seed(42)

# Global parameters
n_samples = 300
input_dim = 2

# Uniformly distributed inputs
lb = 0 * np.ones(input_dim)
ub = 2 * np.ones(input_dim)

def inputs_uniform(n_samples, n_pars, lb, ub):
    return np.vstack(
        np.array([np.random.uniform(lb[i], ub[i], n_samples) for i in range(n_pars)]).T)

# Compute gradients with GPy
def eval_gp_grad(x, f, n_samples, input_dim):
    kernel = GPy.kern.RBF(input_dim=input_dim, ARD=True)
    gp = GPy.models.GPRegression(x, f.reshape(-1, 1), kernel)
    gp.optimize()
    df = gp.predict_jacobian(x)[0].reshape(n_samples, input_dim)
    return df

# Generate and normalize inputs
X = inputs_uniform(n_samples, input_dim, lb, ub)
nor = Normalizer(lb, ub)
x = nor.fit_transform(X)

# Define the output of interest and compute the gradients
func = sin_2d
dfunc = egrad(func)

f = func(x)
df_exact = dfunc(x) # exact gradients
df_gp = eval_gp_grad(x, f, n_samples, input_dim) # gradients approximated with Gaussian process regression
title = "sin"

# Compute the active subspace with approximated gradients
asub1 = ActiveSubspaces(dim=1, method='exact', n_boot=100)
asub1.fit(gradients=df_gp)
asub1.plot_eigenvalues(figsize=(6, 4), title=title)
asub1.plot_eigenvectors(figsize=(6, 4), title=title)
asub1.plot_sufficient_summary(x, f, figsize=(6, 4), title=title)

asub2 = ActiveSubspaces(dim=2, method='exact', n_boot=100)
asub2.fit(gradients=df_gp)
asub2.plot_sufficient_summary(x, f, figsize=(6, 4), title=title)

# Compute the active subspace with exact gradients
asub1_ = ActiveSubspaces(dim=1, method='exact', n_boot=100)
asub1_.fit(gradients=df_exact)
asub1_.plot_eigenvalues(figsize=(6, 4), title=title)
asub1_.plot_eigenvectors(figsize=(6, 4), title=title)
asub1_.plot_sufficient_summary(x, f, figsize=(6, 4), title=title)

asub2_ = ActiveSubspaces(dim=2, method='exact', n_boot=100)
asub2_.fit(gradients=df_exact)
asub2_.plot_sufficient_summary(x, f, figsize=(6, 4), title=title)

# Error analysis
samples_lis = [500+100*i for i in range(6)]
err_abs, err_rel = [], []
for n_samples in samples_lis:
    X = inputs_uniform(n_samples, input_dim, lb, ub)
    nor = Normalizer(lb, ub)
    x = nor.fit_transform(X)
    f = func(x)
    df_exact = dfunc(x)
    df_gp = eval_gp_grad(x, f, *x.shape)
    absdiff = (1/n_samples)*np.sum(np.linalg.norm(df_exact - df_gp, axis=1))
    err_abs += [absdiff]
    err_rel += [absdiff / ((1/n_samples)*np.sum(np.linalg.norm(df_exact, axis=1)))]

fig, axes = plt.subplots(2, 1, sharex=True, figsize=(9, 7))
fig.suptitle('Error Analysis')
axes[0].scatter(samples_lis, err_abs)
axes[0].plot(samples_lis, err_abs)
axes[0].set_ylabel('Absolute error')
axes[1].scatter(samples_lis, err_rel)
axes[1].plot(samples_lis, err_rel)
axes[1].set_ylabel('Relative error')
for i in range(2):
    axes[i].set_yscale('log')
    axes[i].grid(linestyle='dotted')
plt.xlabel('N samples')
plt.show()


# # Approximation with local linear gradients

# Global parameters
n_samples = 300
input_dim = 2

# Generate and normalize inputs
X = inputs_uniform(n_samples, input_dim, lb, ub)
nor = Normalizer(lb, ub)
x = nor.fit_transform(X)

# Define the output of interest and compute the gradients
func = partial(sin_2d, normalizer=nor)
dfunc = egrad(func)

f = func(x)
df_exact = dfunc(x)
title = "sin"

# Compute the active subspace with local_linear_gradients
asub1 = ActiveSubspaces(dim=1, method='local', n_boot=100)
asub1.fit(inputs=x, outputs=f)
asub1.plot_eigenvalues(figsize=(6, 4), title=title)
asub1.plot_eigenvectors(figsize=(6, 4), title=title)
asub1.plot_sufficient_summary(x, f, figsize=(6, 4), title=title)

asub2 = ActiveSubspaces(dim=2, method='local', n_boot=100)
asub2.fit(inputs=x, outputs=f)
asub2.plot_sufficient_summary(x, f, figsize=(6, 4), title=title)

# Compute the active subspace with exact gradients
asub1_ = ActiveSubspaces(dim=1, method='exact', n_boot=100)
asub1_.fit(gradients=df_exact)
asub1_.plot_eigenvalues(figsize=(6, 4), title=title)
asub1_.plot_eigenvectors(figsize=(6, 4), title=title)
asub1_.plot_sufficient_summary(x, f, figsize=(6, 4), title=title)

asub2_ = ActiveSubspaces(dim=2, method='exact', n_boot=100)
asub2_.fit(gradients=df_exact)
asub2_.plot_sufficient_summary(x, f, figsize=(6, 4), title=title)

# Error analysis
samples_lis = [2**i for i in range(6, 12)]
err_abs, err_rel = [], []
for n_samples in samples_lis:
    X = inputs_uniform(n_samples, input_dim, lb, ub)
    nor = Normalizer(lb, ub)
    x = nor.fit_transform(X)
    f = func(x)
    ll_gradients, new_inputs = local_linear_gradients(inputs=x, outputs=f)
    df = egrad(func)(new_inputs)
    absdiff = (1/n_samples)*np.sum(np.linalg.norm(df - ll_gradients, axis=1))
    err_abs += [absdiff]
    err_rel += [absdiff / ((1/n_samples)*np.sum(np.linalg.norm(df, axis=1)))]

fig, axes = plt.subplots(2, 1, sharex=True, figsize=(9, 7))
fig.suptitle('Error Analysis')
axes[0].scatter(samples_lis, err_abs)
axes[0].plot(samples_lis, err_abs)
axes[0].set_ylabel('Absolute error')
axes[1].scatter(samples_lis, err_rel)
axes[1].plot(samples_lis, err_rel)
axes[1].set_ylabel('Relative error')
for i in range(2):
    axes[i].set_yscale('log')
    axes[i].grid(linestyle='dotted')
plt.xlabel('N samples')
plt.show()

