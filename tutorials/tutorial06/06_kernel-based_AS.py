
import autograd.numpy as np
from autograd import elementwise_grad as egrad
import matplotlib.pyplot as plt
from functools import partial
import GPy

from athena.active import ActiveSubspaces
from athena.kas import KernelActiveSubspaces
from athena.feature_map import FeatureMap, rff_map, rff_jac
from athena.projection_factory import ProjectionFactory
from athena.utils import Normalizer, CrossValidation, average_rrmse

from data.numpy_functions import radial


np.random.seed(42)

# global parameters
n_samples = 800  # this is the number of data points to use for the tuning of kas
N = 500 # this is the number of test samples to use
input_dim = 2

# set the dimension of the discrete feature space (D in the introduction)
n_features = 1000

def sample_in_out(input_dim, n_samples):
    #input ranges
    lb = np.array(-3 * np.ones(input_dim))
    ub = np.array(3 * np.ones(input_dim))

    #input normalization
    XX = np.random.uniform(lb, ub, (n_samples, input_dim))

    nor = Normalizer(lb, ub)
    x = nor.fit_transform(XX)

    #output values (f) and gradients (df)
    func = partial(radial, normalizer=nor, generatrix=lambda x: np.cos(x))
    f = func(x)
    df = egrad(func)(x)
    return x, f, df

xx, f, df = sample_in_out(input_dim, n_samples)
y, t, dt = sample_in_out(input_dim, N)


#AS
ss = ActiveSubspaces(1)
ss.fit(gradients=dt, outputs=t, inputs=y)
ss.plot_eigenvalues()
ss.plot_sufficient_summary(y, t)

# number of parameters of the spectral distribution associated to the feature map
# this is the number of parameters to tune after
n_params = 1

# sample the bias term
b = np.random.uniform(0, 2 * np.pi, n_features)

# define the feature map
fm = FeatureMap(distr='laplace',
                bias=b,
                input_dim=input_dim,
                n_features=n_features,
                params=np.zeros(n_params),
                sigma_f=f.var())

# instantiate a KernelActiveSubspaces object with associated feature map 
kss = KernelActiveSubspaces(feature_map=fm, dim=1, n_features=n_features)

# number of folds for the cross-validation algorithm
folds = 3

# Skip if bias and projection matrix are loaded
csv = CrossValidation(inputs=xx,
                      outputs=f.reshape(-1, 1),
                      gradients=df.reshape(n_samples, 1, input_dim),
                      folds=folds,
                      subspace=kss)

best = fm.tune_pr_matrix(func=average_rrmse,
                  bounds=[slice(-2, 1, 0.2) for i in range(n_params)],
                  args=(csv, ),
                  method='bso',
                  maxiter=20,
                  save_file=False)

print('The lowest rrmse is {}%'.format(best[0]))


W = np.load('opt_pr_matrix.npy')
b = np.load('bias.npy')
fm._pr_matrix = W
fm.bias = b
kss.fit(gradients=dt.reshape(N, 1, input_dim),
            outputs=t,
            inputs=y)

kss.plot_eigenvalues(n_evals=5)
kss.plot_sufficient_summary(xx, f)

