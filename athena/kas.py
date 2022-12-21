"""
Module for Kernel-based Active Subspaces.

:References:

    - Francesco Romor, Marco Tezzele, Andrea Lario, Gianluigi Rozza.
      Kernel-based active subspaces with application to computational fluid
      dynamics parametric problems using discontinuous Galerkin method.
      International Journal for Numerical Methods in Engineering,
      123(23):6000–6027, 2022. doi:10.1002/nme.7099

"""
import numpy as np
from .subspaces import Subspaces
from .utils import (initialize_weights, local_linear_gradients)
from .feature_map import FeatureMap


class KernelActiveSubspaces(Subspaces):
    """
    Compute the kernel-based active subspaces given the inputs and the
    gradients of the model function wrt the input parameters, or given the
    input/outputs couples. Only two methods are available: 'exact' and 'local'.

    :param `FeatureMap` feature_map: `athena.feature_map.FeatureMap` object. See
        :class:`FeatureMap` class in :py:mod:`feature_map` module.
    :param int n_features: dimension of the feature space.
    :param str method: method to compute the AS. Possible choices are
        'exact' when the gradients are provided, or 'local' to use local linear
        models. This approach is related to the sufficient dimension reduction
        method known sometimes as the outer product of gradient method. See the
        2001 paper 'Structure adaptive approach for dimension reduction' from
        Hristache, et al.
    :param int n_boot: number of bootstrap samples. Default is None.

    :cvar numpy.ndarray inputs: input parameters oriented as rows.
    :cvar numpy.ndarray outputs: corresponding outputs oriented as rows.
    :cvar numpy.ndarray gradients: n_samples-by-n_params matrix containing the
        gradient samples oriented as rows.
    :cvar numpy.ndarray weights: n_samples-by-1 weight vector, corresponds
        to numerical quadrature rule used to estimate matrix whose eigenspaces
        define the active subspace.
    :cvar int n_features: dimension of the feature space.
    :cvar numpy.ndarray features: n_samples-by-n_features matrix containing
        the projections of the inputs to the n_features-dimensional feature
        space.
    :cvar numpy.ndarray pseudo_gradients:
    :cvar str method: method to compute the AS. Possible choices are
        'exact' when the gradients are provided, or 'local' to use local linear
        models. This approach is related to the sufficient dimension reduction
        method known sometimes as the outer product of gradient method. See the
        2001 paper 'Structure adaptive approach for dimension reduction' from
        Hristache, et al.
    :cvar int n_boot: number of bootstrap samples.
    :cvar numpy.ndarray metric: metric matrix for vectorial active
        subspaces.
    """
    def __init__(self,
                 dim,
                 feature_map=None,
                 n_features=None,
                 method='exact',
                 n_boot=None):
        super().__init__(dim, method, n_boot)
        self.feature_map = feature_map
        self.n_features = n_features
        self.features = None
        self.pseudo_gradients = None

    def fit(self,
            inputs=None,
            outputs=None,
            gradients=None,
            weights=None,
            metric=None):
        """
        Compute the kernel based active subspaces given the inputs and the
        gradients of the model function wrt the input parameters, or given the
        input/outputs couples. Only two methods are available: 'exact' and
        'local'.

        :param numpy.ndarray inputs: array n_samples-by-n_params containing
            the points in the original parameter space.
        :param numpy.ndarray outputs: array n_samples-by-1 containing
            the values of the model function.
        :param numpy.ndarray gradients: array n_samples-by-n_params containing
            the gradient samples oriented as rows.
        :param numpy.ndarray weights: n_samples-by-1 weight vector, corresponds
            to numerical quadrature rule used to estimate matrix whose
            eigenspaces define the active subspace.
        :param numpy.ndarray metric: metric matrix output_dim-by-output-dim for
            vectorial active subspaces.
        :raises: TypeError, ValueError
        """
        if self.method == 'exact':
            if gradients is None or inputs is None:
                raise TypeError('gradients or inputs argument is None.')

        # estimate active subspace with local linear models.
        elif self.method == 'local':
            if inputs is None or outputs is None:
                raise TypeError('inputs or outputs argument is None.')
            gradients, inputs = local_linear_gradients(inputs=inputs,
                                                       outputs=outputs,
                                                       weights=weights)

        if len(gradients.shape) == 2:
            gradients = gradients.reshape(gradients.shape[0], 1,
                                          gradients.shape[1])

        if weights is None or self.method == 'local':
            # use the new gradients to compute the weights, otherwise dimension
            # mismatch accours.
            weights = initialize_weights(gradients)

        if self.n_features is None:
            self.n_features = inputs.shape[1]

        if self.feature_map is None:
            # default spectral measure is Gaussian
            self.feature_map = FeatureMap(distr='multivariate_normal',
                                          bias=np.ones((1, self.n_features)),
                                          input_dim=inputs.shape[1],
                                          n_features=self.n_features,
                                          params=np.ones(inputs.shape[1]),
                                          sigma_f=1)

        if len(gradients.shape) == 3 and metric is None:
            metric = np.diag(np.ones(gradients.shape[1]))

        self.pseudo_gradients, self.features = self._reparametrize(
            inputs, gradients)

        self.evals, self.evects = self._build_decompose_cov_matrix(
            self.pseudo_gradients, weights, metric)

        if self.n_boot:
            if self.n_boot <= 50:
                self._compute_bootstrap_ranges(gradients=self.pseudo_gradients,
                                               weights=weights,
                                               metric=metric)
            else:
                raise ValueError(
                    'n_boot is too high for the bootstrap method applied to kas'
                )

        self._partition()

    def transform(self, inputs):
        """
        Map full variables to active and inactive variables.
        Points in the original input space are mapped to the active and
        inactive non-linear subspace.

        :param numpy.ndarray inputs: array n_samples-by-n_params containing
            the points in the original parameter space.
        :return: array n_samples-by-active_dim containing the mapped active
            variables;
            array n_samples-by-inactive_dim containing the mapped inactive
            variables.
        :rtype: numpy.ndarray, numpy.ndarray
        :raises: TypeError
        """
        features = self.feature_map.compute_fmap(inputs)

        if self.W1 is None:
            raise TypeError(
                'the kernel-based active subspace has not been evaluated.')

        # allow evaluation of active variables only
        if self.W2 is None:
            active = np.dot(features, self.W1)
            inactive = None
        else:
            active = np.dot(features, self.W1)
            inactive = np.dot(features, self.W2)

        return active, inactive

    def inverse_transform(self, reduced_inputs, n_points):
        pass

    def _reparametrize(self, inputs, gradients):
        """
        Computes the pseudo-gradients solving an overdetermined linear system.

        :param numpy.ndarray inputs: array n_samples-by-n_params containing
            the points in the original parameter space.
        :param numpy.ndarray gradients: array n_samples-by-n_params containing
            the gradient samples oriented as rows.
        :return: array n_samples-by-output_dim-by-n_params matrix containing
            the pseudo gradients corresponding to each sample;
            array n_samples-by-n_features containing the image of the inputs
            in the feature space.
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
