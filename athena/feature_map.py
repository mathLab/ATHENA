"""
Module for the feature map class.
"""
import numpy as np
from scipy.optimize import dual_annealing
from .projection_factory import ProjectionFactory


class FeatureMap(object):
    """
    Feature map class.
    TO DOC
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
        return self.distr(self.input_dim, self.n_features, self.params)

    def compute_fmap(self, inputs):
        if self._pr_matrix is None:
            self._pr_matrix = self._compute_pr_matrix()
        return self.fmap(inputs, self._pr_matrix, self.bias, self.n_features,
                         self.sigma_f)

    def compute_fmap_jac(self, inputs):
        if self._pr_matrix is None:
            self._pr_matrix = self._compute_pr_matrix()
        return self.fmap_jac(inputs, self._pr_matrix, self.bias,
                             self.n_features, self.sigma_f)

    def tune_pr_matrix(self, func, bounds, maxiter=30):
        """
        Dual Annealing optimization to tune the parameters of the projection
        matrix.
        A traditional Generalized Simulated Annealing will be performed with no
        local search strategy applied.

        :param callable func: the objective function to be minimized.
            Must be in the form f(x, *args), where x is the argument in the
            form of a 1-D array and args is a tuple of any additional fixed
            parameters needed to completely specify the function.
        :param sequence bounds: shape (n, 2). Bounds for variables.
            (min, max) pairs for each element in x, defining bounds for the
            objective function parameter.
        :param int maxiter: the maximum number of global search iterations.
            Default value is 30.
        :return: the optimization result represented as a OptimizeResult
            object. Important attributes are: `x` the solution array, `fun`
            the value of the function at the solution, and `message` which
            describes the cause of the termination.
        :rtype: scipy.OptimizeResult
        """
        opt_res = dual_annealing(func=func, bounds=bounds, maxiter=maxiter, no_local_search=True)
        self.params = opt_res.x
        self._pr_matrix = self._compute_pr_matrix()

def rff_map(inputs, pr_matrix, bias, n_features, sigma_f):
    """
    Random Fourier Features
    TO DOC
    """
    return np.sqrt(
        4 / n_features) * sigma_f * np.cos(np.dot(inputs, pr_matrix.T) + bias)


def rff_jac(inputs, pr_matrix, bias, n_features, sigma_f):
    """
    Random Fourier Features jacobian
    TO DOC
    """
    return (np.sqrt(2 / n_features) * sigma_f *
            (-1) * np.sin(np.dot(inputs, pr_matrix.T) + bias)).reshape(
                inputs.shape[0], n_features, 1) * pr_matrix
