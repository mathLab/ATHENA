import torch
import GPy
import math
import seaborn as sns
import matplotlib.pyplot as plt
import pyro
from pyro.infer.mcmc import MCMC, NUTS
import pyro.optim

from athena.active import ActiveSubspaces

from torch_functions import sin_2d, radial

import warnings
warnings.filterwarnings('ignore')

# Global parameters
n_samples = 100
input_dim = 5


# Next we define the Likelihood (model) and the surrogate model (surrogate_model) built with active subspaces.
def model(f):
    """
    Likelihood p(f|x), the prior on the inputs is a multivariate Gaussian distribution.
    The model function of interest is f(X)=(X+1)^{2} where 1 is a vector of ones of the dimension of X.
    """
    x = pyro.sample(
        "input",
        pyro.distributions.MultivariateNormal(torch.zeros([input_dim]), torch.eye(input_dim)))
    mean = torch.norm(x + torch.ones(input_dim))**2
    eps = 0.1
    pyro.sample("outputs", pyro.distributions.Normal(mean, eps), obs=f)


def surrogate_model(f, gp):
    """
    Likelihood p(g|s), the prior on the inputs is a multivariate Gaussian distribution.
    The model function of interest is the response function g(S) designed with active subspaces.
    """
    y = pyro.sample("input", pyro.distributions.Normal(0, 1))
    mean = gp.predict(y.cpu().detach().numpy().reshape(-1, 1))[0]
    eps = 0.1
    pyro.sample("outputs", pyro.distributions.Normal(torch.Tensor(mean), eps), obs=f)

#generate inputs, outputs, gradients
dist_inputs = pyro.distributions.MultivariateNormal(torch.zeros([input_dim]),
                                                    torch.eye(input_dim))
x = dist_inputs(sample_shape=torch.Size([n_samples]))
x.requires_grad = True
f = radial(x + torch.ones(input_dim), generatrix=lambda x: x)
f.backward(gradient=torch.ones([n_samples]))
df = x.grad

#search for an active subspace
ss = ActiveSubspaces(dim=1)
ss.fit(gradients=df.cpu().detach().numpy())
ss.plot_eigenvalues(figsize=(6, 4))
ss.plot_sufficient_summary(x.detach().numpy(), f.detach().numpy(), figsize=(6, 4))

kernel = GPy.kern.RBF(input_dim=1, ARD=True)
gp = GPy.models.GPRegression(
    ss.transform(x.detach().numpy())[0],
    f.reshape(-1, 1).detach().numpy(), kernel)
gp.optimize_restarts(5, verbose=False)


# Use No U-Turn Sampler (NUTS) Hamiltonian Monte Carlo to sample from the posterior of the original model.
#plain NUTS
num_chains = 1
num_samples = 100
kernel = NUTS(model)
mcmc = MCMC(kernel,
            num_samples=num_samples,
            warmup_steps=100,
            num_chains=num_chains)
mcmc.run(f)
mcmc.summary()
mcmc_samples = mcmc.get_samples(group_by_chain=True)
print(mcmc_samples.keys())
chains = mcmc_samples["input"]
print(chains.shape)


# Show the probablity posterior distribution of each inputs' component (input_dim).
for i in range(5):
    plt.figure(figsize=(6,4))
    sns.distplot(mcmc_samples['input'][:, :, i])
    plt.title("Full model")
    plt.xlabel("input {}th-component".format(i+1))
    plt.show()


# Posterior samples of the active variable from original model
print(ss.transform(chains[0])[0].mean())
plt.figure(figsize=(6,4))
sns.distplot(ss.transform(chains[0])[0])
plt.title("Full model")
plt.xlabel("active component".format(i+1))
plt.show()


# Use No U-Turn Sampler (NUTS) Hamiltonian Monte Carlo to sample from the posterior of the original model.
#AS NUTS
skernel = NUTS(surrogate_model)
smcmc = MCMC(skernel,
             num_samples=num_samples,
             warmup_steps=100,
             num_chains=num_chains)
smcmc.run(f, gp)
smcmc.summary()

smcmc_samples = smcmc.get_samples(group_by_chain=True)
print(smcmc_samples.keys())
chains = smcmc_samples["input"]
print(chains.shape)


# Show the probablity posterior distribution of the only (active) component.
print(chains[0].mean())
plt.figure(figsize=(6,4))
sns.distplot(smcmc_samples['input'])
plt.title("Surrogate model")
plt.xlabel("input's active variable")
plt.show()
