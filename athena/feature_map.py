"""
Module for the feature map class.
"""
import numpy as np
from scipy.optimize import brute, dual_annealing
from .projection_factory import ProjectionFactory


class FeatureMap(object):
    """
    Feature map class.
    TO DOC
    :param array_like params:
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

    def tune_pr_matrix(self, func, bounds, args=(), method=None, maxiter=50):
        """
        TO DOC
        ADD EXAMPLE for bounds

        :param callable func: the objective function to be minimized.
            Must be in the form f(x, *args), where x is the argument in the
            form of a 1-D array and args is a tuple of any additional fixed
            parameters needed to completely specify the function.
        :param tuple bounds: each component of the bounds tuple must be a
            slice tuple of the form (low, high, step). It defines bounds for
            the objective function parameter. Step will be ignored for
            'dual_annealing' method.
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
        :raises: ValueError
        """
        if method is None:
            if len(self.params) < 4:
                method = 'brute'
            else:
                method = 'dual_annealing'

        if method == 'brute':
            self.params = brute(func=func,
                                ranges=bounds,
                                args=args,
                                finish=None)
        elif method == 'dual_annealing':
            bounds_list = [[bound.start, bound.stop] for bound in bounds]
            self.params = dual_annealing(func=func,
                                         bounds=bounds_list,
                                         args=args,
                                         maxiter=maxiter,
                                         no_local_search=True).x
        else:
            raise ValueError(
                "Method argument can only be 'brute' or 'dual_annealing'.")
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
