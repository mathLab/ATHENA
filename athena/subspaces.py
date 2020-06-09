"""Base module for Active Subspaces and Non-linear Active Subspaces.
"""
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})


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
        
        param numpy.ndarray gradients: M-by-m matrix of gradient samples.
        param numpy.ndarray weights: M-by-1 vector of weights.
        param int nboot: number of bootstrap samples. Default is 100.
        return: array e_br is a m-by-2 matrix, first column contains bootstrap
            lower bound on eigenvalues, second column contains bootstrap upper
            bound on eigenvalues; array sub_br is a (m-1)-by-3 matrix, first
            column contains bootstrap lower bound on estimated subspace error,
            second column contains estimated mean of subspace error (a
            reasonable subspace error estimate), third column contains
            estimated upper bound on subspace error.
        rtype: numpy.ndarray, numpy.ndarray
        """
        n_pars = gradients.shape[1]
        e_boot = np.zeros((n_pars, nboot))
        sub_dist = np.zeros((n_pars - 1, nboot))

        for i in range(nboot):
            gradients0, weights0 = self._bootstrap_replicate(gradients, weights)
            __, e0, W0 = self._build_decompose_cov_matrix(gradients=gradients0,
                                                          weights=weights0,
                                                          method=method)
            e_boot[:, i] = e0.reshape((n_pars, ))
            for j in range(n_pars - 1):
                sub_dist[j, i] = np.linalg.norm(np.dot(self.evects[:, :j + 1].T,
                                                       W0[:, j + 1:]),
                                                ord=2)

        # bootstrap ranges for the eigenvalues
        self.evals_br = np.hstack((np.amin(e_boot, axis=1).reshape(
            (n_pars, 1)), np.amax(e_boot, axis=1).reshape((n_pars, 1))))
        # bootstrap ranges and mean for subspace distance
        self.subs_br = np.hstack((np.amin(sub_dist, axis=1).reshape(
            (n_pars - 1, 1)), np.mean(sub_dist, axis=1).reshape(
                (n_pars - 1, 1)), np.amax(sub_dist, axis=1).reshape(
                    (n_pars - 1, 1))))

    @staticmethod
    def _bootstrap_replicate(matrix, weights):
        """
        Return a bootstrap replicate.
    
        A bootstrap replicate is a sampling-with-replacement strategy from a
        given data set. 
        """
        M = weights.shape[0]
        ind = np.random.randint(M, size=(M, ))
        return matrix[ind, :].copy(), weights[ind, :].copy()

    @classmethod
    def _build_decompose_cov_matrix(cls, *args, **kwargs):
        """
        Not implemented, it has to be implemented in subclasses.
        """
        raise NotImplementedError(
            'Subclass must implement abstract method {}._build_decompose_cov_matrix'
            .format(cls.__name__))

    def compute(self, *args, **kwargs):
        """
        Abstract method to compute the active subspaces.
        Not implemented, it has to be implemented in subclasses.
        """
        raise NotImplementedError(
            'Subclass must implement abstract method {}.compute'.format(
                self.__class__.__name__))

    def forward(self, inputs):
        """
        Abstract method to map full variables to active and inactive variables.
        
        Points in the original input space are mapped to the active and
        inactive subspace.
        
        :param numpy.ndarray inputs: array n_samples-by-n_params containing
            the points in the original parameter space.
        :return: array n_samples-by-active_dim containing the mapped active
            variables; array n_samples-by-inactive_dim containing the mapped
            inactive variables.
        :rtype: numpy.ndarray, numpy.ndarray
        """
        raise NotImplementedError(
            'Subclass must implement abstract method {}.compute'.format(
                self.__class__.__name__))

    def backward(self, reduced_inputs, n_points):
        """
        Abstract method to find points in full space that map to reduced
        variable points.
        Not implemented, it has to be implemented in subclasses.
        
        :param numpy.ndarray reduced_inputs: n_samples-by-n_params matrix that
            contains points in the space of active variables.
        :param int n_points: the number of points in the original parameter
            space that are returned that map to the given active variables.
        """
        raise NotImplementedError(
            'Subclass must implement abstract method {}.backward'.format(
                self.__class__.__name__))

    def partition(self, dim):
        """
        Partition the eigenvectors to define the active and inactive subspaces.
        
        :param int dim: dimension of the active subspace.
        :raises: TypeError, ValueError
        """
        if not isinstance(dim, int):
            raise TypeError('dim should be an integer.')

        if dim < 1 or dim > self.evects.shape[0]:
            raise ValueError(
                'dim must be positive and less than the dimension of the ' \
                ' eigenvectors: dim = {}.'.format(dim))
        self.dim = dim
        self.W1 = self.evects[:, :dim]
        self.W2 = self.evects[:, dim:]

    def plot_eigenvalues(self,
                         n_evals=None,
                         filename=None,
                         figsize=(8, 8),
                         title=''):
        """
        Plot the eigenvalues.
        
        :param int n_evals: number of eigenvalues to plot. If not provided
            all the eigenvalues will be plotted.
        :param str filename: if specified, the plot is saved at `filename`.
        :param tuple(int,int) figsize: tuple in inches defining the figure
            size. Default is (8, 8).
        :param str title: title of the plot.
        :raises: ValueError

        .. warning::
            `self.compute` has to be called in advance.
        """
        if self.evals is None:
            raise ValueError('The eigenvalues have not been computed.'
                             'You have to perform the compute method.')
        if n_evals is None:
            n_evals = self.evals.shape[0]
        if n_evals > self.evals.shape[0]:
            raise ValueError('Invalid number of eigenvalues to plot.')

        plt.figure(figsize=figsize)
        plt.title(title)
        plt.semilogy(range(1, n_evals + 1),
                     self.evals[:n_evals],
                     'ko-',
                     markersize=8,
                     linewidth=2)
        plt.xticks(range(1, n_evals + 1))
        plt.xlabel('Index')
        plt.ylabel('Eigenvalues')
        plt.grid(linestyle='dotted')
        if self.evals_br is None:
            plt.axis([
                0, n_evals + 1, 0.1 * np.amin(self.evals[:n_evals]),
                10 * np.amax(self.evals[:n_evals])
            ])
        else:
            plt.fill_between(range(1, n_evals + 1),
                             self.evals_br[:n_evals, 0],
                             self.evals_br[:n_evals, 1],
                             facecolor='0.7',
                             interpolate=True)
            plt.axis([
                0, n_evals + 1, 0.1 * np.amin(self.evals_br[:n_evals, 0]),
                10 * np.amax(self.evals_br[:n_evals, 1])
            ])

        if filename:
            plt.savefig(filename)
        else:
            plt.show()

    def plot_eigenvectors(self,
                          n_evects=None,
                          filename=None,
                          figsize=None,
                          labels=None,
                          title=''):
        """
        Plot the eigenvectors.
        
        :param int n_evects: number of eigenvectors to plot.
             Default is self.dim.
        :param str filename: if specified, the plot is saved at `filename`.
        :param tuple(int,int) figsize: tuple in inches defining the figure
            size. Default is (8, 2 * n_evects).
        :param str labels: labels for the components of the eigenvectors.
        :param str title: title of the plot.
        :raises: ValueError

        .. warning::
            `self.compute` has to be called in advance.
        """
        if self.evects is None:
            raise ValueError('The eigenvectors have not been computed.'
                             'You have to perform the compute method.')
        if n_evects is None:
            n_evects = self.dim
        if n_evects > self.evects.shape[0]:
            raise ValueError('Invalid number of eigenvectors to plot.')

        if figsize is None:
            figsize = (8, 2 * n_evects)

        n_pars = self.evects.shape[0]
        fig, axes = plt.subplots(n_evects, 1, figsize=figsize)
        fig.suptitle(title)
        # to ensure generality for subplots (1, 1)
        axes = np.array(axes)
        for i, ax in enumerate(axes.flat):
            ax.scatter(range(1, n_pars + 1),
                       self.evects[:n_pars + 1, i],
                       c='blue',
                       s=60,
                       alpha=0.9,
                       edgecolors='k')
            ax.axhline(linewidth=0.7, color='black')

            ax.set_xticks(range(1, n_pars + 1))
            if labels:
                ax.set_xticklabels(labels)

            ax.set_ylabel('Active eigenvector {}'.format(i + 1))
            ax.grid(linestyle='dotted')
            ax.axis([0, n_pars + 1, -1, 1])

        axes.flat[-1].set_xlabel('Eigenvector components')
        fig.tight_layout()
        # tight_layout does not consider suptitle so we need to adjust it manually
        plt.subplots_adjust(top=0.94)

        if filename:
            plt.savefig(filename)
        else:
            plt.show()

    def plot_sufficient_summary(self,
                                inputs,
                                outputs,
                                filename=None,
                                figsize=(10, 8),
                                title=''):
        """
        Plot the sufficient summary.
        
        :param numpy.ndarray inputs: array n_samples-by-n_params containing
            the points in the full input space.
        :param numpy.ndarray outputs: array n_samples-by-1 containing the
            corresponding function evaluations.
        :param str filename: if specified, the plot is saved at `filename`.
        :param tuple(int,int) figsize: tuple in inches defining the figure
            size. Defaults to (10, 8).
        :param str title: title of the plot.
        :raises: ValueError

        .. warning::
            `self.partition` has to be called in advance. 

            Plot only available for partitions up to dimension 2.
        """
        if self.dim is None:
            raise ValueError('You first have to partition your subspaces.')

        plt.figure(figsize=figsize)
        plt.title(title)

        if self.dim == 1:
            plt.scatter(inputs.dot(self.W1),
                        outputs,
                        c='blue',
                        s=40,
                        alpha=0.9,
                        edgecolors='k')
            plt.xlabel('Active variable ' + r'$W_1^T \mathbf{\mu}}$',
                       fontsize=18)
            plt.ylabel(r'$f \, (\mathbf{\mu})$', fontsize=18)
        elif self.dim == 2:
            x = inputs.dot(self.W1)
            plt.scatter(x[:, 0],
                        x[:, 1],
                        c=outputs.reshape(-1),
                        s=60,
                        alpha=0.9,
                        edgecolors='k',
                        vmin=np.min(outputs),
                        vmax=np.max(outputs))
            plt.xlabel('First active variable', fontsize=18)
            plt.ylabel('Second active variable', fontsize=18)
            ymin = 1.1 * np.amin([np.amin(x[:, 0]), np.amin(x[:, 1])])
            ymax = 1.1 * np.amax([np.amax(x[:, 0]), np.amax(x[:, 1])])
            plt.axis('equal')
            plt.axis([ymin, ymax, ymin, ymax])
            plt.colorbar()
        else:
            raise ValueError(
                'Sufficient summary plots cannot be made in more than 2 ' \
                'dimensions.'
            )

        plt.grid(linestyle='dotted')

        if filename:
            plt.savefig(filename)
        else:
            plt.show()
