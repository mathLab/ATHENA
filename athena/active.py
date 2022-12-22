"""
Active Subspaces module.

:References:

    - Paul Constantine. Active subspaces: Emerging ideas for dimension
      reduction in parameter studies, vol. 2 of SIAM Spotlights, SIAM, 2015.

    - Constantine et al. Python Active-subspaces Utility Library,
      Journal of Open Source Software, 1(5), 79, 2016.

"""
import types
import numpy as np
from scipy.linalg import null_space

from .subspaces import Subspaces
from .utils import (Normalizer, initialize_weights, linear_program_ineq,
                    local_linear_gradients)


class ActiveSubspaces(Subspaces):
    """Active Subspaces class

    :param int dim: dimension of the active subspace.
    :param str method: method to compute the AS. Possible choices are
        'exact' when the gradients are provided, or 'local' to use local linear
        models. This approach is related to the sufficient dimension reduction
        method known sometimes as the outer product of gradient method. See the
        2001 paper 'Structure adaptive approach for dimension reduction' from
        Hristache, et al.
    :param int n_boot: number of bootstrap samples. Default is 100.
    """
    def __init__(self, dim, method='exact', n_boot=100):
        super().__init__(dim, method, n_boot)

    @property
    def activity_scores(self):
        """
        Return the activity scores as defined in Constantine and Diaz, Global
        sensitivity metrics from active subspaces, arxiv.org/abs/1510.04361
        Equation (21).

        :return: array with the activity score of each parameter.
        :rtype: numpy.ndarray
        :raises: TypeError

        .. warning:: `self.fit` has to be called in advance.
        """
        if self.W1 is None:
            raise TypeError('The eigenvectors have not been computed and '
                            'partitioned. You have to perform the fit method.')

        return np.power(self.W1, 2).dot(self.evals[:self.dim])

    def fit(self,
            inputs=None,
            outputs=None,
            gradients=None,
            weights=None,
            metric=None):
        """
        Compute the active subspaces given the gradients of the model function
        wrt the input parameters, or given the input/outputs couples. Only two
        methods are available: 'exact' and 'local'.

        :param numpy.ndarray inputs: input parameters oriented as rows.
        :param numpy.ndarray outputs: corresponding outputs oriented as rows.
        :param numpy.ndarray gradients: n_samples-by-n_params matrix containing
            the gradient samples oriented as rows. If frequent directions needed
            to be performed, gradients is an object of GeneratorType.
        :param numpy.ndarray weights: n_samples-by-1 weight vector, corresponds
            to numerical quadrature rule used to estimate matrix whose
            eigenspaces define the active subspace.
        :param numpy.ndarray metric: metric matrix output_dim-by-output-dim for
            vectorial active subspaces.
        :raises: TypeError

        :Example:

            >>> # inputs shape is n_samples-by-n_params
            >>> # outputs shape is n_samples-by-1
            >>> # gradients shape is n_samples-by-n_params

            >>> # if gradients are available use the 'exact' method:
            >>> ss1 =  ActiveSubspaces(dim=1, method='exact', n_boot=150)
            >>> ss1.fit(gradients=gradients)

            >>> # for the frequent direction method to compute the eigenpairs
            >>> # you need to pass a generator to gradients:
            >>> gradients_gen = (grad for grad in gradients)
            >>> ss2 =  ActiveSubspaces(dim=2, method='exact')
            >>> ss2.fit(gradients=gradients_gen)

            >>> # if no gradients are available use the 'local' method:
            >>> ss3 =  ActiveSubspaces(dim=1, method='local', n_boot=150)
            >>> ss3.fit(inputs=inputs, outputs=outputs)
        """
        if self.method == 'exact' and gradients is None:
            raise TypeError('gradients argument is None.')

        # estimate active subspace with local linear models.
        if self.method == 'local':
            if inputs is None or outputs is None:
                raise TypeError('inputs or outputs argument is None.')
            gradients = local_linear_gradients(inputs=inputs,
                                               outputs=outputs,
                                               weights=weights)[0]

        if isinstance(gradients, types.GeneratorType):
            self.evals, self.evects = self._frequent_directions(gradients)
        else:
            if weights is None or self.method == 'local':
                # use the new gradients to compute the weights, otherwise
                # dimension mismatch accours.
                weights = initialize_weights(gradients)

            if len(gradients.shape) == 3 and metric is None:
                metric = np.diag(np.ones(gradients.shape[1]))

            self.evals, self.evects = self._build_decompose_cov_matrix(
                gradients=gradients, weights=weights, metric=metric)

            self._compute_bootstrap_ranges(gradients, weights, metric=metric)
        self._partition()

    def transform(self, inputs):
        """
        Map full variables to active and inactive variables.

        Points in the original input space are mapped to the active and inactive
        subspace.

        :param numpy.ndarray inputs: array n_samples-by-n_params containing the
            points in the original parameter space.
        :return: array n_samples-by-active_dim containing the mapped active
            variables;
            array n_samples-by-inactive_dim containing the mapped inactive
            variables.
        :rtype: numpy.ndarray, numpy.ndarray
        :raises: TypeError
        """
        if self.W1 is None:
            raise TypeError('the active subspace has not been evaluated.')

        active = np.dot(inputs, self.W1)
        # allow evaluation of active variables only
        inactive = None if self.W2 is None else np.dot(inputs, self.W2)
        return active, inactive

    def inverse_transform(self, reduced_inputs, n_points=1):
        """
        Map the points in the active variable space to the original parameter
        space.

        :param numpy.ndarray reduced_inputs: n_samples-by-dim matrix that
            contains points in the space of active variables.
        :param int n_points: the number of points in the original parameter
            space that are returned that map to the given active variables.
            Defaults to 1.
        :return: (n_samples * n_points)-by-n_params matrix that contains
            points in the original parameter space, (n_samples *
            n_points)-by-n_params matrix that contains integer indices. These
            indices identify which rows of the previous matrix (the full
            parameters) map to which rows of the active variables matrix.
        :rtype: numpy.ndarray, numpy.ndarray
        :raises: TypeError

        .. note:: The inverse map depends critically on the
            `self._sample_inactive` method.
        """
        if self.W1 is None:
            raise TypeError('the active subspace has not been evaluated.')

        # the inactive eigenvectors are needed
        if self.W2 is None:
            self.W2 = null_space(self.W1).T

        inactive_swap = np.array([
            self._sample_inactive(red_inp, n_points)
            for red_inp in reduced_inputs
        ])
        inactive_inputs = np.swapaxes(inactive_swap, 1, 2)

        inputs, indices = self._rotate_x(reduced_inputs, inactive_inputs)
        return inputs, indices

    def _sample_inactive(self, reduced_input, n_points):
        """
        Sample inactive variables.

        Sample values of the inactive variables for a fixed value of the active
        variables when the original variables are bounded by a hypercube.

        :param numpy.ndarray reduced_input: the value of the active variables.
        :param int n_points: the number of inactive variable samples,
        :return: n_points-by-(inactive_dim) matrix that contains values of the
            inactive variable that correspond to the given `reduced_input`.
        :rtype: numpy.ndarray

        .. note:: The trick here is to sample the inactive variables z so that
            -1 <= W1*y + W2*z <= 1, where y is the given value of the active
            variables. In other words, we need to sample z such that it respects
            the linear inequalities W2*z <= 1 - W1*y, -W2*z <= 1 + W1*y. These
            inequalities define a polytope in R^(inactive_dim). We want to
            sample N points uniformly from the polytope. This function first
            tries a simple rejection sampling scheme, which finds a bounding
            hyperbox for the polytope, draws points uniformly from the bounding
            hyperbox, and rejects points outside the polytope. If that method
            does not return enough samples, the method tries a "hit and run"
            method for sampling from the polytope. If that does not work, it
            returns an array with `N` copies of a feasible point computed as the
            Chebyshev center of the polytope.
        """
        Z = self._rejection_sampling_inactive(reduced_input, n_points)
        if Z is None:
            Z = self._hit_and_run_inactive(reduced_input, n_points)
        return Z

    def _compute_A_b(self, reduced_input):
        """
        Compute the matrix A and the vector b to build a box around the
        inactive subspace for uniform sampling.

        :param numpy.ndarray reduced_input: the value of the active variables.
        :return: matrix A, and vector b.
        :rtype: numpy.ndarray, numpy.ndarray
        """
        s = np.dot(self.W1, reduced_input).reshape((-1, 1))
        A = np.vstack((self.W2, -1 * self.W2))
        b = np.vstack((-1 - s, -1 + s)).reshape((-1, 1))
        return A, b

    def _rejection_sampling_inactive(self, reduced_input, n_points):
        """
        A rejection sampling method for sampling the from a polytope.

        :param numpy.ndarray reduced_input: the value of the active variables.
        :param int n_points: the number of inactive variable samples.
        :return: n_points-by-inactive_dim matrix that contains values of the
            inactive variable that correspond to the given `reduced_input`.
        :rtype: numpy.ndarray
        """
        m, n = self.W1.shape
        inactive_dim = m - n
        # Build a box around z for uniform sampling
        A, b = self._compute_A_b(reduced_input)
        lbox, ubox = np.zeros((1, inactive_dim)), np.zeros((1, inactive_dim))
        for i in range(inactive_dim):
            clb = np.zeros((inactive_dim, 1))
            clb[i, 0] = 1.0
            lbox[0, i] = linear_program_ineq(clb, A, b)[i, 0]
            cub = np.zeros((inactive_dim, 1))
            cub[i, 0] = -1.0
            ubox[0, i] = linear_program_ineq(cub, A, b)[i, 0]
        bn = Normalizer(lbox, ubox)
        Zbox = bn.inverse_transform(
            np.random.uniform(-1.0, 1.0, size=(50 * n_points, inactive_dim)))
        ind = np.all(np.dot(A, Zbox.T) >= b, axis=0)

        if np.sum(ind) >= n_points:
            Z = Zbox[ind, :]
            return Z[:n_points, :].reshape(n_points, inactive_dim)
        return None

    def _hit_and_run_inactive(self, reduced_input, n_points):
        """
        A hit and run method for sampling the inactive variables from a
        polytope.

        :param numpy.ndarray reduced_input: the value of the active variables.
        :param int n_points: the number of inactive variable samples.
        :return: n_points-by-(inactive_dim) matrix that contains values of the
            inactive variable that correspond to the given `reduced_input`.
        :rtype: numpy.ndarray
        """
        m, n = self.W1.shape
        inactive_dim = m - n

        # Get an initial feasible point using the Chebyshev center. Huge props
        # to David Gleich for the Chebyshev center.
        s = np.dot(self.W1, reduced_input).reshape((m, 1))
        normW2 = np.sqrt(np.sum(np.power(self.W2, 2), axis=1)).reshape((m, 1))
        A = np.hstack((np.vstack(
            (self.W2, -self.W2.copy())), np.vstack((normW2, normW2.copy()))))
        b = np.vstack((1 - s, 1 + s)).reshape((2 * m, 1))
        c = np.zeros((inactive_dim + 1, 1))
        c[-1] = -1.0

        zc = linear_program_ineq(c, -1 * A, -b)
        z0 = zc[:-1].reshape((inactive_dim, 1))

        # define the polytope A >= b
        A, b = self._compute_A_b(reduced_input)

        # tolerance
        eps0 = 1e-6 / 4.0

        Z = np.zeros((n_points, inactive_dim))
        for i in range(n_points):
            # random direction
            bad_dir = True
            count, maxcount = 0, 50
            while bad_dir:
                d = np.random.normal(size=(inactive_dim, 1))
                bad_dir = np.any(np.dot(A, z0 + eps0 * d) <= b)
                count += 1
                if count >= maxcount:
                    Z[i:, :] = np.tile(z0, (1, n_points - i)).T
                    return Z

            # find constraints that impose lower and upper bounds on eps
            f, g = b - np.dot(A, z0), np.dot(A, d)

            # find an upper bound on the step
            min_ind = np.logical_and(g <= 0,
                                     f < -np.sqrt(np.finfo(np.float64).eps))
            eps_max = np.amin(f[min_ind] / g[min_ind])

            # find a lower bound on the step
            max_ind = np.logical_and(g > 0,
                                     f < -np.sqrt(np.finfo(np.float64).eps))
            eps_min = np.amax(f[max_ind] / g[max_ind])

            # randomly sample eps
            eps1 = np.random.uniform(eps_min, eps_max)

            # take a step along d
            z1 = z0 + eps1 * d
            Z[i, :] = z1.reshape(-1, )

            # update temp var
            z0 = z1.copy()

        return Z

    def _rotate_x(self, reduced_inputs, inactive_inputs):
        """
        A convenience function for rotating subspace coordinates to x space.

        :param numpy.ndarray reduced_input: the value of the active variables.
        :param numpy.ndarray inactive_inputs: the value of the inactive
            variables.
        :return: (n_samples * n_points)-by-n_params matrix that contains points
            in the original parameter space, (n_samples * n_points)-by-n_params
            matrix that contains integer indices. These indices identify which
            rows of the previous matrix (the full parameters) map to which rows
            of the active variables matrix.
        :rtype: numpy.ndarray, numpy.ndarray
        """
        NY, n = reduced_inputs.shape
        N = inactive_inputs.shape[2]
        m = n + inactive_inputs.shape[1]

        YY = np.tile(reduced_inputs.reshape((NY, n, 1)), (1, 1, N))
        YZ = np.concatenate((YY, inactive_inputs), axis=1).transpose(
            (1, 0, 2)).reshape((m, N * NY)).transpose((1, 0))
        inputs = np.dot(YZ, self.evects.T).reshape((N * NY, m))
        indices = np.kron(np.arange(NY), np.ones(N)).reshape((N * NY, 1))
        return inputs, indices

    def _frequent_directions(self, gradients):
        """
        Function that performs the frequent directions algorithm for
        matrix sketching. For more details about the method, see
        "Frequent directions: Simple and deterministic matrix
        sketching." Ghashami, Mina, et al.
        SIAM Journal on Computing 45.5 (2016): 1762-1792.
        doi: https://doi.org/10.1137/15M1009718

        :param iterable gradients: generator for spatial gradients.
        :return: the sorted eigenvalues and the corresponding eigenvectors for
            the reduced matrix.
        :rtype: numpy.ndarray, numpy.ndarray
        """
        s = np.array([next(gradients) for _ in range(self.dim)]).T
        for grad in gradients:
            evects, sigma = np.linalg.svd(s, full_matrices=False)[:2]
            s = np.dot(
                evects,
                np.sqrt(np.diag(sigma**2) - (sigma[-1]**2) * np.eye(self.dim)))
            s[:, -1] = grad
        evals = sigma**2
        return evals, evects
