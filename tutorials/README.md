# Tutorials

In this folder you can find a collection of useful Notebooks containing several tutorials, in order to understand the principles and the potential of **ATHENA**.

#### [Tutorial 1](01_intro.ipynb)
Here we show a basic application of active subspaces on a simple model in order to reconstruct and analyze it.

#### [Tutorial 2](02_gradients_evaluation.ipynb)
Here we focus on a crucial step of the procedure: the evaluation of the gradients of the model with respect to the inputs. We show two possible methods to approximate the gradients given pairs of input-output datasets: Gaussian process regression, which requires [GPy](https://github.com/SheffieldML/GPy) and local linear gradients, which is implemented in ATHENA. In [Tutorial 5](05_SPDE_on_athena_vectorial_AS.ipynb) we will use adjoint methods to reconstruct the gradients.

#### [Tutorial 3](03_response_surfaces.ipynb)
Here we exploit the presence of an active subspace to build one-dimensional response surfaces with Gaussian
processes. We compare the choice of the original model as profile of the ridge approximation with the choice of the optimal profile. It requires [GPy](https://github.com/SheffieldML/GPy) and [pyhmc](https://github.com/rmcgibbo/pyhmc) for Hamiltonian Monte Carlo.

#### [Tutorial 4](04_inverse_problems.ipynb)
Here we show an application of the active subspaces property to speed up the sampling from the posterior of an inverse problem with Gaussian prior and likelihood. We use the library [Pyro](https://pyro.ai/) for probabilistic programming and Hamiltonian Monte Carlo and [GPy](https://github.com/SheffieldML/GPy) for Gaussian process regression.

#### [Tutorial 5](05_SPDE_on_athena_vectorial_AS.ipynb)
You need to run [Tutorial 5, solver](05_SPDE_on_fenics_solver.ipynb) first. Here we show how an
active subspace can be searched for in the case of a model with vectorial outputs. We use [fenics](https://fenicsproject.org/) to solve a Poisson problem with red noise in the diffusion coefficient (approximated with truncated Karhunen-Loève decomposition). If you want to look at the active eigenvectors and K-L modes after having ran the tutorial, open [Tutorial 5, visualization tool](05_SPDE_on_fenics_modes.ipynb).

#### More to come...
We plan to add more tutorials but the time is often against us. If you want to contribute with a notebook on a feature not covered yet we will be very happy and give you support on editing!

The main references for these tutorials are

* [Constantine, Paul G. active subspaces: emerging ideas for dimension reduction in parameter studies](https://doi.org/10.1137/1.9781611973860),

* [Zahm, Olivier, Paul G. Constantine, Clementine Prieur, and Youssef M. Marzouk. "Gradient-based dimension reduction of multivariate vector-valued functions."](https://epubs.siam.org/doi/pdf/10.1137/18M1221837).
