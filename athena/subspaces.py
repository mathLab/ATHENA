"""[summary]

[description]
"""
import numpy as np
import matplotlib.pyplot as plt

class Subspaces(object):
    """Active Subspaces base class
    
    [description]
    """
    def __init__(self):
        self.W1 = None
        self.W2 = None
        self.evals = None
        self.evects = None
        self.evals_br = None
        self.subs_br = None
        self.dim = None
        self.cov_matrix = None

    def _compute_bootstrap_ranges(self, gradients, weights, method, nboot=100):
        """Compute bootstrap ranges for eigenvalues and subspaces.
        
        An implementation of the nonparametric bootstrap that we use in 
        conjunction with the subspace estimation methods to estimate the errors in 
        the eigenvalues and subspaces.
        
        Parameters
        ----------
        gradients : ndarray
            M-by-m matrix of gradient samples
        weights : ndarray
            M-by-1 vector of weights corresponding to samples
        nboot : int, optional
            number of bootstrap samples (default 100)
            
        Returns
        -------
        e_br : ndarray
            m-by-2 matrix, first column contains bootstrap lower bound on 
            eigenvalues, second column contains bootstrap upper bound on 
            eigenvalues
        sub_br : ndarray
            (m-1)-by-3 matrix, first column contains bootstrap lower bound on 
            estimated subspace error, second column contains estimated mean of
            subspace error (a reasonable subspace error estimate), third column
            contains estimated upper bound on subspace error    
        """
        n_pars = gradients.shape[1]
        e_boot = np.zeros((n_pars, nboot))
        sub_dist = np.zeros((n_pars-1, nboot))
        
        for i in range(nboot):
            gradients0, weights0 = self._bootstrap_replicate(gradients, weights)
            __, e0, W0 = self._build_decompose_cov_matrix(gradients=gradients0, weights=weights0, method=method)
            e_boot[:, i] = e0.reshape((n_pars, ))
            for j in range(n_pars-1):
                sub_dist[j, i] = np.linalg.norm(np.dot(self.evects[:, :j+1].T, W0[:, j+1:]), ord=2)
        
        # bootstrap ranges for the eigenvalues
        self.evals_br = np.hstack((np.amin(e_boot, axis=1).reshape((n_pars, 1)), np.amax(e_boot, axis=1).reshape((n_pars, 1))))
        # bootstrap ranges and mean for subspace distance
        self.subs_br = np.hstack((np.amin(sub_dist, axis=1).reshape((n_pars-1, 1)), np.mean(sub_dist, axis=1).reshape((n_pars-1, 1)), np.amax(sub_dist, axis=1).reshape((n_pars-1, 1))))

    @staticmethod
    def _bootstrap_replicate(matrix, weights):
        """Return a bootstrap replicate.
    
        A bootstrap replicate is a sampling-with-replacement strategy from a given
        data set. 
        """
        M = weights.shape[0]
        ind = np.random.randint(M, size=(M, ))
        return matrix[ind, :].copy(), weights[ind, :].copy()

    @classmethod
    def _build_decompose_cov_matrix(cls, *args, **kwargs):
        """
        """
        raise NotImplementedError(
            'Subclass must implement abstract method {}._build_decompose_cov_matrix'.format(
                cls.__name__))

    def compute(self, *args, **kwargs):
        """
        Abstract method to compute ...
        Not implemented, it has to be implemented in subclasses.
        """
        raise NotImplementedError(
            'Subclass must implement abstract method {}.compute'.format(
                self.__class__.__name__))

    def forward(self, inputs):
        """Map full variables to active variables.
        
        Map the points in the original input space to the active and inactive
        variables.
        Parameters
        ----------
        inputs : ndarray
            an M-by-m matrix, each row of `X` is a point in the original 
            parameter space
        Returns
        -------
        active : ndarray 
            M-by-n matrix that contains points in the space of active variables.
            Each row of `Y` corresponds to a row of `X`.
        inactive : ndarray 
            M-by-(m-n) matrix that contains points in the space of inactive 
            variables. Each row of `Z` corresponds to a row of `X`.
        """
        active = np.dot(inputs, self.W1)
        inactive = np.dot(inputs, self.W2)
        return active, inactive

    def backward(self, reduced_inputs, n_points):
        """
        Abstract method to find points in full space that map to reduced variable points.
        Not implemented, it has to be implemented in subclasses.
        
        Map the points in the active variable space to the original parameter
        space.
        
        Parameters
        ----------
        reduced_inputs : ndarray
            M-by-n matrix that contains points in the space of active variables
        n_points : int, optional
            the number of points in the original parameter space that are 
            returned that map to the given active variables (default 1)

        """
        raise NotImplementedError(
            'Subclass must implement abstract method {}.backward'.format(
                self.__class__.__name__))

    def partition(self, dim):
        """Partition the eigenvectors to define the active subspace.
        
        A convenience function for partitioning the full set of eigenvectors to
        separate the active from inactive subspaces.
        """
        if not isinstance(dim, int):
            raise TypeError('dim should be an integer')

        if dim < 1 or dim > self.evects.shape[0]:
            raise ValueError('dim ({}) must be positive and less than the dimension of the eigenvectors.'.format(dim))

        self.W1 = self.evects[:, :dim]
        self.W2 = self.evects[:, dim:]

    def plot_eigenvalues(self, figsize=(8, 8), title=''):
        """
        Plot the eigenvalues.
    
        :param tuple(int,int) figsize: tuple in inches defining the figure
            size. Default is (8, 8).
        :param str title: title of the plot.
        """
        if self.evals is None:
            raise ValueError('The eigenvalues have not been computed.'
                             'You have to perform the compute method.')
        n_pars = self.evals.shape[0]
        plt.figure(figsize=figsize)
        plt.title(title)
        plt.semilogy(range(1, n_pars+1), self.evals, 'ko-', markersize=8, linewidth=2)
        plt.xticks(range(1, n_pars+1))
        plt.xlabel('Index')
        plt.ylabel('Eigenvalues')
        plt.grid(linestyle='dotted')
        if self.evals_br is None:
            plt.axis([0, n_pars+1, 0.1*np.amin(self.evals), 10*np.amax(self.evals)])
        else:
            plt.fill_between(range(1, n_pars+1), self.evals_br[:, 0], self.evals_br[:, 1], facecolor='0.7', interpolate=True)
            plt.axis([0, n_pars+1, 0.1*np.amin(self.evals_br[:, 0]), 10*np.amax(self.evals_br[:, 1])])
        plt.show()


