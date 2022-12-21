"""
Module for the feature map class

:References:

    - Francesco Romor, Marco Tezzele, Andrea Lario, Gianluigi Rozza.
      Kernel-based active subspaces with application to computational fluid
      dynamics parametric problems using discontinuous Galerkin method.
      International Journal for Numerical Methods in Engineering,
      123(23):6000–6027, 2022. doi:10.1002/nme.7099

"""
from functools import partial
import numpy as np
from scipy.optimize import brute, dual_annealing
import GPyOpt
from .projection_factory import ProjectionFactory


class FeatureMap():
    """Feature map class.

    :param str distr: name of the spectral distribution to sample from the
        matrix.
    :param numpy.ndarray bias: n_features dimensional bias. It is sampled from a
        Unifrom distribution on the interval [0, 2*PI].
    :param int input_dim: dimension of the input space.
    :param int n_features: dimension of the Reproducing Kernel Hilbert Space.
    :param list params: number of hyperparameters of the spectral distribution.
    :param int sigma_f: multiplicative constant of the feature
        map. Usually it is set as the empirical variance of the outputs.

    :cvar callable fmap: feature map used to project the input samples to the RKHS.
        Default value is rff_map.
    :cvar callable fjac: Jacobian matrix of fmap. Default value is rff_jac, the
        analytical Jacobian of fmap.
    :cvar numpy.ndarray _pr_matrix: n_features-by-input_dim projection
        matrix, whose rows are sampled from the spectral distribution distr.

    :raises TypeError
    """
    def __init__(self, distr, bias, input_dim, n_features, params, sigma_f):
        if callable(distr):
            self.distr = distr
        elif isinstance(distr, str):
            self.distr = ProjectionFactory(distr)
        else:
            raise TypeError('`distr` is not valid.')

        self.bias = bias
        self.input_dim = input_dim
        self.n_features = n_features
        self.params = params
        self.sigma_f = sigma_f
        self.fmap = rff_map
        self.fmap_jac = rff_jac

        self._pr_matrix = None

    @property
    def pr_matrix(self):
        """
        Get the projection matrix.

        :return: the projection matrix.
        :rtype: numpy.ndarray
        """
        return self._pr_matrix

    def _compute_pr_matrix(self):
        """
        Sample the projection matrixx from the spectral distribution.

        :return: the projection matrix.
        :rtype: numpy.ndarray
        """
        return self.distr(self.input_dim, self.n_features, self.params)

    def compute_fmap(self, inputs):
        """
        Evaluate the feature map at inputs.

        :param numpy.ndarray inputs: the inputs to project on the RKHS.
        :return: the n_features dimensional projection of the inputs.
        :rtype: numpy.ndarray
        """
        if self._pr_matrix is None:
            self._pr_matrix = self._compute_pr_matrix()
        return self.fmap(inputs, self._pr_matrix, self.bias, self.n_features,
                         self.sigma_f)

    def compute_fmap_jac(self, inputs):
        """
        Evaluate the Jacobian matrix of feature map at inputs.

        :param numpy.ndarray inputs: the inputs at which compute the Jacobian
            matrix of the feature map.
        :return: the n_features-by-input_dim dimensional Jacobian of the
            feature map at the inputs.
        :rtype: numpy.ndarray
        """
        if self._pr_matrix is None:
            self._pr_matrix = self._compute_pr_matrix()
        return self.fmap_jac(inputs, self._pr_matrix, self.bias,
                             self.n_features, self.sigma_f)

    def tune_pr_matrix(self,
                       func,
                       bounds,
                       method=None,
                       maxiter=50,
                       save_file=False,
                       fn_args=None):
        """
        Tune the parameters of the spectral distribution. Three methods are
        available: log-grid-search (brute), annealing (dual_annealing) and
        Bayesian stochastic optimization (bso) from GpyOpt. The default object
        function to optimize is athena.utils.average_rrmse, which uses a
        cross-validation procedure from athena.utils, see Example and tutorial 06_kernel-based_AS.

        :Example:
        >>> from athena.kas import KernelActiveSubspaces
        >>> from athena.feature_map import FeatureMap
        >>> from athena.projection_factory import ProjectionFactory
        >>> from athena.utils import CrossValidation, average_rrmse
        >>> input_dim, output_dim, n_samples = 2, 1, 30
        >>> n_features, n_params = 10, 1
        >>> xx = np.ones((n_samples, input_dim))
        >>> f = np.ones((n_samples, output_dim))
        >>> df = np.ones((n_samples, output_dim, input_dim))
        >>> fm = FeatureMap(distr='laplace',
                            bias=np.random.uniform(0, 2 * np.pi, n_features),
                            input_dim=input_dim,
                            n_features=n_features,
                            params=np.zeros(n_params),
                            sigma_f=f.var())
        >>> kss = KernelActiveSubspaces(feature_map=fm, dim=1, n_features=n_features)
        >>> csv = CrossValidation(inputs=xx,
                                outputs=f.reshape(-1, 1),
                                gradients=df.reshape(n_samples, output_dim, input_dim),
                                folds=3,
                                subspace=kss)
        >>> best = fm.tune_pr_matrix(func=average_rrmse,
                                    bounds=[slice(-2, 1, 0.2) for i in range(n_params)],
                                    args=(csv, ),
                                    method='bso',
                                    maxiter=20,
                                    save_file=False)

        :param callable func: the objective function to be minimized.
            Must be in the form f(x, *args), where x is the argument in the
            form of a 1-D array and args is a tuple of any additional fixed
            parameters needed to completely specify the function.
        :param tuple bounds: each component of the bounds tuple must be a
            slice tuple of the form (low, high, step). It defines bounds for
            the objective function parameter in a logarithmic scale. Step will
            be ignored for 'dual_annealing' method.
        :param tuple args: any additional fixed parameters needed to
            completely specify the objective function.
        :param str method: method used to optimize the objective function.
            Possible values are 'brute', or 'dual_annealing'.
            Default value is None, and the choice is made automatically wrt
            the dimension of `self.params`. If the dimension is less than 4
            brute force is used, otherwise a traditional Generalized
            Simulated Annealing will be performed with no local search
            strategy applied.
        :param int maxiter: the maximum number of global search iterations.
            Default value is 50.
        :param bool save_file: True to save the optimal projection matrix
        :param dict fn_args: dictionary of arguments passed to func.
        :raises: ValueError
        :return: list that records the best score and the best projection
            matrix. The initial values are 0.8 and a n_features-by-input_dim
            numpy.ndarray of zeros.
        :rtype: list
        """
        if fn_args is None:
            fn_args = {}
        best = [0.8, np.zeros((self.n_features, self.input_dim))]

        if method is None:
            method = 'brute' if len(self.params) < 4 else 'dual_annealing'
        if method == 'brute':
            self.params = brute(func=func,
                                ranges=bounds,
                                args=(
                                    best,
                                    *tuple(fn_args.values()),
                                ),
                                finish=None)
        elif method == 'dual_annealing':
            bounds_list = [[bound.start, bound.stop] for bound in bounds]
            self.params = 10**dual_annealing(func=func,
                                             bounds=bounds_list,
                                             args=(
                                                 best,
                                                 *tuple(fn_args.values()),
                                             ),
                                             maxiter=maxiter,
                                             no_local_search=False).x
        elif method == 'bso':
            bounds = [{
                'name': f'var_{str(i)}',
                'type': 'continuous',
                'domain': [bound.start, bound.stop],
            } for i, bound in enumerate(bounds)]
            func_obj = partial(func, best=best, **fn_args)
            bopt = GPyOpt.methods.BayesianOptimization(func_obj,
                                                       domain=bounds,
                                                       model_type='GP',
                                                       acquisition_type='EI',
                                                       exact_feval=True)
            bopt.run_optimization(max_iter=maxiter,
                                  max_time=3600,
                                  eps=1e-16,
                                  verbosity=False)
            self.params = 10**bopt.x_opt
        else:
            raise ValueError(
                "Method argument can only be 'brute' or 'dual_annealing' or 'bso'."
            )

        self._pr_matrix = best[1]
        self.params = best[0]

        if save_file:
            np.save("opt_pr_matrix", best[1])

        return best


def rff_map(inputs, pr_matrix, bias, n_features, sigma_f):
    """
    Random Fourier Features map. It can be vectorized for inputs of shape n_samples-by-input_dim.

    :param numpy.ndarray inputs: input_dim dimensional inputs to project to the RKHS.
    :param numpy.ndarray pr_matrix: n_features-by-input_dim projection matrix,
        whose rows are sampled from the spectral distribution.
    :param numpy.ndarray bias: n_features dimensional bias. It is sampled from a
        Unifrom distribution on the interval [0, 2*PI].
    :param int n_features: dimension of the RKHS
    :param int sigma_f: multiplicative term representing the empirical variance
        the outptus.
    :return: n_features dimensional projection of the inputs to the RKHS
    :rtype: numpy.ndarray
    """
    return np.sqrt(2 / n_features) * sigma_f * np.cos(
        np.dot(inputs, pr_matrix.T) + bias.reshape(1, -1))


def rff_jac(inputs, pr_matrix, bias, n_features, sigma_f):
    """
    Random Fourier Features map's Jacobian. It can be vectorized for inputs of shape n_samples-by-input_dim.

    :param numpy.ndarray inputs: input_dim dimensional inputs to project to the RKHS.
    :param numpy.ndarray pr_matrix: n_features-by-input_dim projection matrix,
        whose rows are sampled from the spectral distribution.
    :param numpy.ndarray bias: n_features dimensional bias. It is sampled from a
        Unifrom distribution on the interval [0, 2*PI].
    :param int n_features: dimension of the RKHS
    :param int sigma_f: multiplicative term representing the empirical variance
        the outptus.
    :return: n_features-by-input_dim dimensional projection of the inputs to the RKHS
    :rtype: numpy.ndarray
    """
    return (np.sqrt(2 / n_features) * sigma_f *
            (-1) * np.sin(np.dot(inputs, pr_matrix.T) + bias)).reshape(
                inputs.shape[0], n_features, 1) * pr_matrix
