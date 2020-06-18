"""
Module for the feature map class.
"""
import numpy as np
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
        if self._pr_matrix == None:
            self._pr_matrix = self._compute_pr_matrix()
        return self.fmap(inputs, self._pr_matrix, self.bias, self.n_features,
                         self.sigma_f)

    def compute_fmap_jac(self, inputs):
        if self._pr_matrix == None:
            self._pr_matrix = self._compute_pr_matrix()
        return self.fmap_jac(inputs, self._pr_matrix, self.bias,
                             self.n_features, self.sigma_f)


def hadamard(M, W):
    """
    (nfeatures, (nfeatures, m))
    TO DOC
    """
    return M.reshape(-1, 1) * W


def rff_map(inputs, pr_matrix, bias, n_features, sigma_f):
    """
    Random Fourier Features
    TO DOC
    """
    return np.sqrt(
        4 / n_features) * sigma_f * np.cos(np.dot(pr_matrix, inputs) + bias)


def rff_jac(inputs, pr_matrix, bias, n_features, sigma_f):
    """
    Random Fourier Features jacobian
    TO DOC
    """
    return hadamard(
        np.sqrt(2 / n_features) * sigma_f * (-1) *
        np.sin(np.dot(pr_matrix, inputs) + bias), pr_matrix)
