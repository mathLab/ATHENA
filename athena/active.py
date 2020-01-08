"""[summary]

[description]
"""
import numpy as np
from .subspaces import Subspaces
from .utils import Normalizer, initialize_weights, linear_program_ineq, local_linear_gradients, sort_eigpairs


class ActiveSubspaces(Subspaces):
    """Active Subspaces base class
    
    [description]
    """
    def __init__(self):
        super().__init__()

    @staticmethod
    def _build_decompose_cov_matrix(inputs=None,
                                    outputs=None,
                                    gradients=None,
                                    weights=None,
                                    method=None):
        if method == 'exact' or method == 'local':
            cov_matrix = gradients.T.dot(gradients * weights)
            evals, evects = sort_eigpairs(cov_matrix)
        return cov_matrix, evals, evects

    def compute(self,
                inputs=None,
                outputs=None,
                gradients=None,
                weights=None,
                method='exact',
                nboot=100):
        """[summary]
        
        Parameters
        ----------
        gradients : ndarray
            M-by-m matrix containing the gradient samples oriented as rows
        weights : ndarray
            M-by-1 weight vector, corresponds to numerical quadrature rule used to
            estimate matrix whose eigenspaces define the active subspace

        Local linear models: This approach is related to the sufficient dimension reduction method known 
        sometimes as the outer product of gradient method. See the 2001 paper 
        'Structure adaptive approach for dimension reduction' from Hristache, et al.    
        """
        if method == 'exact':
            if gradients is None:
                raise ValueError('gradients argument is None.')
            if weights is None:
                # default weights is for Monte Carlo
                weights = initialize_weights(gradients)
            self.cov_matrix, self.evals, self.evects = self._build_decompose_cov_matrix(
                gradients=gradients, weights=weights, method=method)
            self._compute_bootstrap_ranges(gradients,
                                           weights,
                                           method=method,
                                           nboot=nboot)

        # estimate active subspace with local linear models.
        if method == 'local':
            if inputs is None or outputs is None:
                raise ValueError('inputs or outputs argument is None.')
            gradients = local_linear_gradients(inputs=inputs,
                                               outputs=outputs,
                                               weights=weights)
            if weights is None:
                # use the new gradients to compute the weights, otherwise dimension mismatch accours.
                weights = initialize_weights(gradients)
            self.cov_matrix, self.evals, self.evects = self._build_decompose_cov_matrix(
                gradients=gradients, weights=weights, method=method)
            self._compute_bootstrap_ranges(gradients,
                                           weights,
                                           method=method,
                                           nboot=nboot)

    def backward(self, reduced_inputs, n_points=1):
        """
        Map the points in the active variable space to the original parameter
        space.
        
        :param numpy.ndarray reduced_inputs: n_samples-by-n_params matrix that
            contains points in the space of active variables.
        :param int n_points: the number of points in the original parameter
            space that are returned that map to the given active variables.
            Default is set to 1.
        
        :return: (n_samples * n_points)-by-n_params matrix that contains points
            in the original parameter space, 
            (n_samples * n_points)-by-n_params matrix that contains integer
            indices. These indices identify which rows of the previous matrix
            (the full parameters) map to which rows of the active variables
            matrix.
        :rtype: numpy.ndarray, numpy.ndarray

        Notes
        -----
        The inverse map depends critically on the `regularize_z` function.
        """
        inactive_swap = np.array([
            self._sample_inactive(red_inp, n_points)
            for red_inp in reduced_inputs
        ])
        inactive_inputs = np.swapaxes(inactive_swap, 1, 2)

        inputs, indices = self._rotate_x(reduced_inputs, inactive_inputs)
        return inputs, indices

    def _sample_inactive(self, reduced_input, n_points):
        """Sample inactive variables.
        
        Sample values of the inactive variables for a fixed value of the active
        variables when the original variables are bounded by a hypercube.
        Parameters
        ----------
        n_points: int 
            the number of inactive variable samples
        reduced_input : ndarray 
            the value of the active variables
        Returns
        -------
        Z : ndarray
            n_points-by-(m-n) matrix that contains values of the inactive variable that 
            correspond to the given `y`
        Notes
        -----
        The trick here is to sample the inactive variables z so that
        -1 <= W1*y + W2*z <= 1,
        where y is the given value of the active variables. In other words, we need
        to sample z such that it respects the linear equalities
        W2*z <= 1 - W1*y, -W2*z <= 1 + W1*y.
        These inequalities define a polytope in R^(m-n). We want to sample `N`
        points uniformly from the polytope.
        This function first tries a simple rejection sampling scheme, which (i)
        finds a bounding hyperbox for the polytope, (ii) draws points uniformly from
        the bounding hyperbox, and (iii) rejects points outside the polytope.
        If that method does not return enough samples, the method tries a "hit and
        run" method for sampling from the polytope.
        If that doesn't work, it returns an array with `N` copies of a feasible
        point computed as the Chebyshev center of the polytope.
        """
        Z = self._rejection_sampling_inactive(reduced_input, n_points)
        if Z is None:
            Z = self._hit_and_run_inactive(reduced_input, n_points)
        return Z

    def _rejection_sampling_inactive(self, reduced_input, n_points):
        """A rejection sampling method for sampling the from a polytope.
        Parameters
        ----------
        N : int 
            the number of inactive variable samples
        y : ndarray 
            the value of the active variables
        W1 : ndarray 
            m-by-n matrix that contains the eigenvector bases of the n-dimensional 
            active subspace
        W2 : ndarray 
            m-by-(m-n) matrix that contains the eigenvector bases of the 
            (m-n)-dimensional inactive subspace
        Returns
        -------
        Z : ndarray
            N-by-(m-n) matrix that contains values of the inactive variable that 
            correspond to the given `y`    
        
        See Also
        --------
        domains.sample_z
        
        Notes
        -----
        The interface for this implementation is written specifically for 
        `domains.sample_z`.
        """
        m, n = self.W1.shape
        inactive_dim = m - n
        s = np.dot(self.W1, reduced_input).reshape((m, 1))

        # Build a box around z for uniform sampling
        A = np.vstack((self.W2, -self.W2))
        b = np.vstack((-1 - s, -1 + s)).reshape((2 * m, 1))
        lbox, ubox = np.zeros((1, m - n)), np.zeros((1, inactive_dim))
        for i in range(inactive_dim):
            clb = np.zeros((inactive_dim, 1))
            clb[i, 0] = 1.0
            lbox[0, i] = linear_program_ineq(clb, A, b)[i, 0]
            cub = np.zeros((inactive_dim, 1))
            cub[i, 0] = -1.0
            ubox[0, i] = linear_program_ineq(cub, A, b)[i, 0]
        bn = Normalizer(lbox, ubox)
        Zbox = bn.unnormalize(
            np.random.uniform(-1.0, 1.0, size=(50 * n_points, inactive_dim)))
        ind = np.all(np.dot(A, Zbox.T) >= b, axis=0)

        if np.sum(ind) >= n_points:
            Z = Zbox[ind, :]
            return Z[:n_points, :].reshape(n_points, inactive_dim)
        else:
            return None

    def _hit_and_run_inactive(self, reduced_input, n_points):
        """A hit and run method for sampling the inactive variables from a polytope.
        Parameters
        ----------
        N : int 
            the number of inactive variable samples
        y : ndarray 
            the value of the active variables
        W1 : ndarray 
            m-by-n matrix that contains the eigenvector bases of the n-dimensional 
            active subspace
        W2 : ndarray 
            m-by-(m-n) matrix that contains the eigenvector bases of the 
            (m-n)-dimensional inactive subspace
        Returns
        -------
        Z : ndarray
            N-by-(m-n) matrix that contains values of the inactive variable that 
            correspond to the given `y`    
        
        See Also
        --------
        domains.sample_z
        
        Notes
        -----
        The interface for this implementation is written specifically for 
        `domains.sample_z`.
        """
        m, n = self.W1.shape
        inactive_dim = m - n

        # get an initial feasible point using the Chebyshev center. huge props to
        # David Gleich for showing me the Chebyshev center.
        s = np.dot(self.W1, reduced_input).reshape((m, 1))
        normW2 = np.sqrt(np.sum(np.power(self.W2, 2), axis=1)).reshape((m, 1))
        A = np.hstack((np.vstack(
            (self.W2, -self.W2.copy())), np.vstack((normW2, normW2.copy()))))
        b = np.vstack((1 - s, 1 + s)).reshape((2 * m, 1))
        c = np.zeros((inactive_dim + 1, 1))
        c[-1] = -1.0

        zc = linear_program_ineq(c, -A, -b)
        z0 = zc[:-1].reshape((inactive_dim, 1))

        # define the polytope A >= b
        s = np.dot(self.W1, reduced_input).reshape((m, 1))
        A = np.vstack((self.W2, -self.W2))
        b = np.vstack((-1 - s, -1 + s)).reshape((2 * m, 1))

        # tolerance
        ztol = 1e-6
        eps0 = ztol / 4.0

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
                                     f < -np.sqrt(np.finfo(np.float).eps))
            eps_max = np.amin(f[min_ind] / g[min_ind])

            # find a lower bound on the step
            max_ind = np.logical_and(g > 0,
                                     f < -np.sqrt(np.finfo(np.float).eps))
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
        """A convenience function for rotating subspace coordinates to x space.
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
