"""
Module for Kernel-based Active Subspaces.

Reference:
- Francesco Romor, Marco Tezzele, Andrea Lario, Gianluigi Rozza.
Kernel-based Active Subspaces with application to CFD problems using
Discontinuous Galerkin Method. 2020. 
arxiv: 
"""
import numpy as np
from .subspaces import Subspaces
from .utils import (initialize_weights, sort_eigpairs, local_linear_gradients)
from .feature_map import FeatureMap


class KernelActiveSubspaces(Subspaces):
    """Kernel Active Subspaces class
    """
    def __init__(self):
        super().__init__()
        self.n_features = None
        self.feature_map = None
        self.features = None
        self.pseudo_gradients = None

    @staticmethod
    def _build_decompose_cov_matrix(gradients=None,
                                    weights=None,
                                    method=None,
                                    metric=None):
        """
        Computes the uncentered covariance matrix of the pseudo_gradients.

        :param numpy.ndarray weights: n_samples-by-1 weight vector, corresponds to numerical
            quadrature rule used to estimate matrix whose eigenspaces define the active
            subspace.
        :param str method: the method used to compute the gradients.
        :param numpy.ndarray metric: output_dim-byoutput-dim the matrix representing the metric
            in the output space
        :return: array n_features containing the eigenvalues ordered decreasingly in magnitude;
            array n_features-by-n_features the columns contain the ordered eigenvectors.
        :rtype: numpy.ndarray, numpy.ndarray, numpy.ndarray
        """
        if method == 'exact' or method == 'local':
            if metric:
                cov_matrix = np.array(
                    np.sum([
                        weights[i, 0] *
                        np.dot(gradients[i, :, :].T,
                               np.dot(metric, gradients[i, :, :]))
                        for i in range(gradients.shape[0])
                    ],
                           axis=0))
                evals, evects = sort_eigpairs(cov_matrix)
                return evals, evects
            else:
                X = np.squeeze(gradients * np.sqrt(weights).reshape(-1, 1, 1))
                _, singular, evects = np.linalg.svd(X, full_matrices=False)
                evals = singular**2
                return evals, evects.T

    def forward(self, inputs):
        """
        Map full variables to active and inactive variables.
        Points in the original input space are mapped to the active and
        inactive non-linear subspace.

        :param numpy.ndarray inputs: array n_samples-by-n_params containing
            the points in the original parameter space.
        :return: array n_samples-by-active_dim containing the mapped active
            variables; array n_samples-by-inactive_dim containing the mapped
            inactive variables.
        :rtype: numpy.ndarray, numpy.ndarray
        """
        features = self.feature_map.compute_fmap(inputs)
        active = np.dot(features, self.W1)
        inactive = np.dot(features, self.W2)
        return active, inactive

    def backward(self, reduced_inputs, n_points):
        pass

    def _reparametrize(self, inputs=None, gradients=None):
        """
        Computes the pseudo-gradients solving an overdetermined linear system.

        :param numpy.ndarray inputs: array n_samples-by-n_params containing
            the points in the original parameter space.
        :param numpy.ndarray gradients: array n_samples-by-n_params containing
            the gradient samples oriented as rows.
        :return: array n_samples-by-n_features containing
            the psuedo gradients; array n_samples-by-n_features containing the
            image of the inputs in the feature space.
        :rtype: numpy.ndarray, numpy.ndarray
        """
        n_samples = inputs.shape[0]

        # Initialize Jacobian for each input
        jacobian = self.feature_map.compute_fmap_jac(inputs)

        # Compute pseudo gradients
        pseudo_gradients = np.array([
            np.linalg.lstsq(jacobian[i, :, :].T,
                            gradients[i, :, :].T,
                            rcond=None)[0].T for i in range(n_samples)
        ])

        # Compute features
        features = self.feature_map.compute_fmap(inputs)

        return pseudo_gradients, features

    def compute(self,
                inputs=None,
                outputs=None,
                gradients=None,
                weights=None,
                method='exact',
                nboot=None,
                n_features=None,
                feature_map=None,
                metric=None):
        """
        Compute the kernel based active subspaces given the inputs and the 
        gradients of the model function wrt the input parameters, or given the input/outputs
        couples. Only two methods are available: 'exact' and 'local'.

        :param numpy.ndarray inputs: array n_samples-by-n_params containing
            the points in the original parameter space.
        :param numpy.ndarray outputs: array n_samples-by-1 containing
            the values of the model function.
        :param numpy.ndarray gradients: array n_samples-by-n_params containing
            the gradient samples oriented as rows.
        :param numpy.ndarray weights: n_samples-by-1 weight vector, corresponds to numerical
            quadrature rule used to estimate matrix whose eigenspaces define the active
            subspace.
        :param str method: the method used to compute the gradients.
        :param int nboot: number of bootstrap samples.
        :param int n_features: dimension of the feature space.
        :param feature_map: feature map object.
        :param numpy.ndarray metric: output_dim-byoutput-dim the matrix representing the metric
            in the output space
        """
        if method == 'exact':
            if gradients is None or inputs is None:
                raise ValueError('gradients or inputs argument is None.')

        # estimate active subspace with local linear models.
        elif method == 'local':
            if inputs is None or outputs is None:
                raise ValueError('inputs or outputs argument is None.')
            gradients = local_linear_gradients(inputs=inputs,
                                               outputs=outputs,
                                               weights=weights).reshape(inputs.shape[0], 1, n_features)

        if weights is None:
            # default weights is for Monte Carlo
            weights = initialize_weights(gradients)

        if n_features is None:
            self.n_features = inputs.shape[1]
        else:
            self.n_features = n_features

        if feature_map is None:
            # default spectral measure is Gaussian
            self.feature_map = FeatureMap(distr='multivariate_normal',
                                          bias=np.ones((1, n_features)),
                                          input_dim=inputs.shape[1],
                                          n_features=n_features,
                                          params=np.ones(inputs.shape[1]),
                                          sigma_f=1)
        else:
            self.feature_map = feature_map

        self.pseudo_gradients, self.features = self._reparametrize(
            inputs, gradients)

        self.evals, self.evects = self._build_decompose_cov_matrix(
            self.pseudo_gradients, weights, method, metric)

        if nboot:
            if nboot <= 50:
                self._compute_bootstrap_ranges(gradients=self.pseudo_gradients,
                                               weights=weights,
                                               method=method,
                                               nboot=nboot)
            else:
                raise ValueError(
                    'the value of nboot is too high for the bootstrap method applied to kas'
                )
