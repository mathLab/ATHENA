"""[summary]

[description]
"""
import numpy as np
from .subspaces import Subspaces
from .utils import initialize_weights, local_linear_gradients, sort_eigpairs


class ActiveSubspaces(Subspaces):
    """Active Subspaces base class
    
    [description]
    """
    def __init__(self):
        super().__init__()

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
