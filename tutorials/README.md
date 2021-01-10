# Tutorials

In this folder you can find a collection of useful Notebooks containing several tutorials to understand the principles and the potential of **ATHENA**.

#### Tutorial 1 [[.ipynb](tutorial01/01_intro.ipynb), [.py](tutorial01/01_intro.py), [.html](https://mathlab.github.io/ATHENA/tutorial1intro.html)]
Here we show a basic application of active subspaces on a simple model in order to reconstruct and analyze it.

#### Tutorial 2 [[.ipynb](tutorial02/02_gradients_evaluation.ipynb), [.py](tutorial02/02_gradients_evaluation.py), [.html](https://mathlab.github.io/ATHENA/tutorial2gradients.html)]
Here we focus on a crucial step of the procedure: the evaluation of the gradients of the model with respect to the inputs. We show two possible methods to approximate the gradients given pairs of input-output datasets: Gaussian process regression, which makes use of [GPy](https://github.com/SheffieldML/GPy), and local linear gradients, which is implemented in ATHENA. In [Tutorial 5](tutorial05/05_SPDE_on_athena_vectorial_AS.ipynb) we will use adjoint methods to reconstruct the gradients.

#### Tutorial 3 [[.ipynb](tutorial03/03_response_surfaces.ipynb), [.py](tutorial03/03_response_surfaces.py), [.html](https://mathlab.github.io/ATHENA/tutorial3response.html)]
Here we exploit the presence of an active subspace to build one-dimensional response surfaces with Gaussian
processes. We compare the choice of the original model as a profile of the ridge approximation with the choice of the optimal profile. It requires [GPy](https://github.com/SheffieldML/GPy) and [pyhmc](https://github.com/rmcgibbo/pyhmc) for Hamiltonian Monte Carlo.

#### [Tutorial 4][[.ipynb](tutorial04/04_inverse_problems.ipynb), [.py](tutorial04/04_inverse_problems.py), [.html](https://mathlab.github.io/ATHENA/tutorial4inverse.html)]
Here we show an application of the active subspaces property to speed up the sampling from the posterior of an inverse problem with Gaussian prior and likelihood. We use the library [Pyro](https://pyro.ai/) for probabilistic programming and Hamiltonian Monte Carlo and [GPy](https://github.com/SheffieldML/GPy) for Gaussian process regression.

#### [Tutorial 5][[.ipynb](tutorial05/05_SPDE_on_athena_vectorial_AS.ipynb), [.py](tutorial05/05_SPDE_on_athena_vectorial_AS.py), [.html](https://mathlab.github.io/ATHENA/tutorial5spde.html)]
You need to run [Tutorial 5, solver](tutorial05/05_SPDE_on_fenics_solver.ipynb) first. Here we show how an
active subspace can be searched for in the case of a model with vectorial outputs. We use [fenics](https://fenicsproject.org/) to solve a Poisson problem with red noise in the diffusion coefficient (approximated with truncated Karhunen-Loève decomposition). If you want to look at the active eigenvectors and K-L modes after having ran the tutorial, open [Tutorial 5, visualization tool](tutorial05/05_SPDE_on_fenics_modes.ipynb).

#### Tutorial 6 [[.ipynb](tutorial06/06_kernel-based_AS.ipynb), [.py](tutorial06/06_kernel-based_AS.py), [.html](https://mathlab.github.io/ATHENA/tutorial6kas.html)]
Here we show how a kernel-based active subspace can be detected and employed
when a standard active subspace is missing. We also describe the tuning
procedure involved.

#### Tutorial 7 [[.ipynb](tutorial07/07_nonlinear_level-set_learning.ipynb), [.py](tutorial07/07_nonlinear_level-set_learning.py), [.html](https://mathlab.github.io/ATHENA/tutorial7nll.html)]
We present the nonlinear level-set learning (NLL) technique and we compare it with AS.

#### More to come...
We plan to add more tutorials but the time is often against us. If you want to contribute with a notebook on a feature not covered yet we will be very happy and give you support on editing!


### References
The main references for these tutorials are

* [Paul G. Constantine. "Active Subspaces: emerging ideas for dimension reduction in parameter studies"](https://doi.org/10.1137/1.9781611973860).

* [Olivier Zahm, Paul G. Constantine, Clementine Prieur, and Youssef M. Marzouk. "Gradient-based dimension reduction of multivariate vector-valued functions."](https://epubs.siam.org/doi/pdf/10.1137/18M1221837).

* [Guannan Zhang, Jiaxin Zhang, and Jacob Hinkle. "Learning nonlinear level sets for dimensionality reduction in function approximation."](https://arxiv.org/abs/1902.10652).
