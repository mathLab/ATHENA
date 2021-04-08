"""
Base module for Active Subspaces and Kernel-based Active Subspaces.

:References:

    - Paul Constantine. Active subspaces: Emerging ideas for dimension
      reduction in parameter studies, vol. 2 of SIAM Spotlights, SIAM, 2015.

    - Francesco Romor, Marco Tezzele, Andrea Lario, Gianluigi Rozza.
      Kernel-based Active Subspaces with application to CFD problems using
      Discontinuous Galerkin Method. 2020.
      arxiv: https://arxiv.org/abs/2008.12083

"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.extmath import randomized_svd
from .utils import sort_eigpairs
plt.rcParams.update({'font.size': 16})


class Subspaces():
    """Active Subspaces base class

    :param str method: method to compute the AS. Possible choices are
        'exact' when the gradients are provided, or 'local' to use local linear
        models. This approach is related to the sufficient dimension reduction
        method known sometimes as the outer product of gradient method. See the
        2001 paper 'Structure adaptive approach for dimension reduction' from
        Hristache, et al.
    :param int n_boot: number of bootstrap samples. Default is 100.
    """
    def __init__(self, dim, method='exact', n_boot=100):
        self.dim = dim
        self.method = method
        self.n_boot = n_boot
        self.W1 = None
        self.W2 = None
        self.evals = None
        self.evects = None
        self.evals_br = None
        self.subs_br = None

    def _build_decompose_cov_matrix(self,
                                    gradients=None,
                                    weights=None,
                                    metric=None):
        """
        Build and decompose the covariance matrix of the gradients.

        :param numpy.ndarray gradients: n_samples-by-n_params matrix containing the
            gradient samples oriented as rows.
        :param numpy.ndarray weights: n_samples-by-1 weight vector, corresponds
            to numerical quadrature rule used to estimate matrix whose eigenspaces
            define the active subspace.
        :param numpy.ndarray metric: metric matrix output_dim-by-output-dim for
            vectorial active subspaces.

        :return: the sorted eigenvalues, and the corresponding eigenvectors.
        :rtype: numpy.ndarray, numpy.ndarray
        """
        if self.method == 'exact' or self.method == 'local':
            if metric is not None:
                cov_matrix = np.array(
                    np.sum([
                        weights[i, 0] *
                        np.dot(gradients[i, :, :].T,
                               np.dot(metric, gradients[i, :, :]))
                        for i in range(gradients.shape[0])
                    ],
                           axis=0))
                evals, evects = sort_eigpairs(cov_matrix)
                return np.squeeze(evals), evects

            X = np.squeeze(gradients * np.sqrt(weights).reshape(-1, 1))
            n_samples, n_pars = X.shape

            # computational complexity of svd and random svd
            svd_complexity = n_samples * n_pars * self.dim
            rsvd_complexity = n_samples * n_pars * np.log(
                self.dim) + (n_samples + n_pars) * self.dim**2

            if svd_complexity > rsvd_complexity and (n_samples > 10000
                                                     or n_pars > 10000):
                singular, evects = randomized_svd(
                    M=X,
                    n_components=self.dim,
                    n_oversamples=10,
                    n_iter='auto',
                    power_iteration_normalizer='auto',
                    transpose='auto')[1:]
            else:
                singular, evects = np.linalg.svd(X, full_matrices=False)[1:]

            evals = singular**2
            return evals, evects.T

    def _compute_bootstrap_ranges(self, gradients, weights, metric=None):
        """Compute bootstrap ranges for eigenvalues and subspaces.

        An implementation of the nonparametric bootstrap that we use in
        conjunction with the subspace estimation methods to estimate the errors
        in the eigenvalues and subspaces.

        :param numpy.ndarray gradients: n_samples-by-n_params matrix containing
            the gradient samples oriented as rows.
        :param numpy.ndarray weights: n_samples-by-1 weight vector, corresponds
            to numerical quadrature rule used to estimate matrix whose
            eigenspaces define the active subspace.
        :param numpy.ndarray metric: metric matrix output_dim-by-output-dim for
            vectorial active subspaces.
        :return: array e_br is a m-by-2 matrix, first column contains
            bootstrap lower bound on eigenvalues, second column contains
            bootstrap upper bound on eigenvalues; array sub_br is a (m-1)-by-3
            matrix, first column contains bootstrap lower bound on estimated
            subspace error, second column contains estimated mean of subspace
            error (a reasonable subspace error estimate), third column contains
            estimated upper bound on subspace error.
        :rtype: numpy.ndarray, numpy.ndarray
        """
        n_pars = gradients.shape[-1]
        n_samples = gradients.shape[0]

        svd_complexity = n_samples * n_pars * self.dim
        rsvd_complexity = n_samples * n_pars * np.log(
            self.dim) + (n_samples + n_pars) * self.dim**2

        # randomized_svd is not implemented for vectorial as yet
        if len(gradients.shape) == 2:
            if svd_complexity > rsvd_complexity and (n_samples > 10000
                                                     or n_pars > 10000):
                range_dim = self.dim
            else:
                range_dim = min(n_pars, n_samples)
        else:
            range_dim = n_pars

        e_boot = np.zeros((range_dim, self.n_boot))
        sub_dist = np.zeros((range_dim - 1, self.n_boot))

        for i in range(self.n_boot):
            gradients0, weights0 = self._bootstrap_replicate(
                gradients, weights)
            e0, W0 = self._build_decompose_cov_matrix(gradients=gradients0,
                                                      weights=weights0,
                                                      metric=metric)
            e_boot[:, i] = e0
            for j in range(range_dim - 1):
                range_diff = np.dot(self.evects[:, :j + 1].T, W0[:, j + 1:])
                sub_dist[j, i] = np.linalg.norm(range_diff, ord=2)

        # bootstrap ranges for the eigenvalues
        self.evals_br = np.hstack((np.amin(e_boot, axis=1).reshape(
            (range_dim, 1)), np.amax(e_boot, axis=1).reshape((range_dim, 1))))
        # bootstrap ranges and mean for subspace distance
        self.subs_br = np.hstack((np.amin(sub_dist, axis=1).reshape(
            (range_dim - 1, 1)), np.mean(sub_dist, axis=1).reshape(
                (range_dim - 1, 1)), np.amax(sub_dist, axis=1).reshape(
                    (range_dim - 1, 1))))

    @staticmethod
    def _bootstrap_replicate(matrix, weights):
        """
        Return a bootstrap replicate.

        A bootstrap replicate is a sampling-with-replacement strategy from a
        given data set.

        :param numpy.ndarray matrix: matrix from which will be sampled N rows. N
            corresponds to the number of rows of weights.
        :param numpy.ndarray weights: n_samples-by-1 weight vector, corresponds
            to numerical quadrature rule used to estimate matrix whose
            eigenspaces define the active subspace.
        """
        ind = np.random.randint(weights.shape[0], size=(weights.shape[0], ))

        # matrix has shape 2 if the outputs are scalar and shape 3 if they are
        # vectorial.
        if len(matrix.shape) == 2:
            return matrix[ind, :].copy(), weights[ind, :].copy()
        elif len(matrix.shape) == 3:
            return matrix[ind, :, :].copy(), weights[ind, :].copy()
        return None, None

    def fit(self, *args, **kwargs):
        """
        Abstract method to compute the active subspaces. Not implemented, it has
        to be implemented in subclasses.
        """
        raise NotImplementedError(
            'Subclass must implement abstract method {}.fit'.format(
                self.__class__.__name__))

    def transform(self, inputs):
        """
        Abstract method to map full variables to active and inactive variables.

        Points in the original input space are mapped to the active and inactive
        subspace.

        :param numpy.ndarray inputs: array n_samples-by-n_params containing the
            points in the original parameter space.
        :return: array n_samples-by-active_dim containing the mapped active variables;
            array n_samples-by-inactive_dim containing the mapped inactive
            variables.
        :rtype: numpy.ndarray, numpy.ndarray
        """
        raise NotImplementedError('Subclass must implement abstract method '
                                  '{}.transform'.format(
                                      self.__class__.__name__))

    def inverse_transform(self, reduced_inputs, n_points):
        """
        Abstract method to find points in full space that map to reduced
        variable points. Not implemented, it has to be implemented in
        subclasses.

        :param numpy.ndarray reduced_inputs: n_samples-by-n_params matrix that
            contains points in the space of active variables.
        :param int n_points: the number of points in the original parameter
            space that are returned that map to the given active variables.
        """
        raise NotImplementedError(
            'Subclass must implement abstract method {}.inverse_transform'.
            format(self.__class__.__name__))

    def _partition(self):
        """
        Partition the eigenvectors to define the active and inactive subspaces.

        :raises: TypeError, ValueError
        """
        if not isinstance(self.dim, int):
            raise TypeError('dim should be an integer.')

        if self.dim < 1 or self.dim > self.evects.shape[1]:
            raise ValueError(
                'dim must be positive and less than the dimension of the '
                ' eigenvectors: dim = {}.'.format(self.dim))

        # allow evaluation of active eigenvectors only
        if self.evects.shape[1] < self.evects.shape[0]:
            self.W1 = self.evects[:, :self.dim]
            self.W2 = None
        elif self.evects.shape[1] == self.evects.shape[0]:
            self.W1 = self.evects[:, :self.dim]
            self.W2 = self.evects[:, self.dim:]
        else:
            raise ValueError(
                'the eigenvectors cannot have dimension less than dim = {}.'.
                format(self.dim))

    def plot_eigenvalues(self,
                         n_evals=None,
                         filename=None,
                         figsize=(8, 8),
                         title=''):
        """
        Plot the eigenvalues.

        :param int n_evals: number of eigenvalues to plot. If not provided all
            the eigenvalues will be plotted.
        :param str filename: if specified, the plot is saved at `filename`.
        :param tuple(int,int) figsize: tuple in inches defining the figure size.
            Default is (8, 8).
        :param str title: title of the plot.
        :raises: TypeError

        .. warning:: `self.fit` has to be called in advance.
        """
        if self.evals is None:
            raise TypeError('The eigenvalues have not been computed.'
                             'You have to perform the fit method.')
        if n_evals is None:
            n_evals = self.evals.shape[0]
        if n_evals > self.evals.shape[0]:
            raise TypeError('Invalid number of eigenvalues to plot.')

        plt.figure(figsize=figsize)
        plt.title(title)
        if np.amin(self.evals[:n_evals]) == 0:
            plt.semilogy(range(1, n_evals + 1),
                         self.evals[:n_evals] + np.finfo(float).eps,
                         'ko-',
                         markersize=8,
                         linewidth=2)
        else:
            plt.semilogy(range(1, n_evals + 1),
                         self.evals[:n_evals],
                         'ko-',
                         markersize=8,
                         linewidth=2)
        plt.xticks(range(1, n_evals + 1))
        plt.xlabel('Index')
        plt.ylabel('Eigenvalues')

        if self.evals_br is None:
            plt.axis([
                0, n_evals + 1, 0.1 * np.amin(self.evals[:n_evals]),
                10 * np.amax(self.evals[:n_evals])
            ])
        else:
            if np.amin(self.evals[:n_evals]) == 0:
                plt.fill_between(
                    range(1, n_evals + 1),
                    self.evals_br[:n_evals, 0] * (1 + np.finfo(float).eps),
                    self.evals_br[:n_evals, 1] * (1 + np.finfo(float).eps),
                    facecolor='0.7',
                    interpolate=True)
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

        plt.grid(linestyle='dotted')
        plt.tight_layout()
        
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

        :param int n_evects: number of eigenvectors to plot. Default is self.dim.
        :param str filename: if specified, the plot is saved at `filename`.
        :param tuple(int,int) figsize: tuple in inches defining the figure size.
            Default is (8, 2 * n_evects).
        :param str labels: labels for the components of the eigenvectors.
        :param str title: title of the plot.
        :raises: ValueError, TypeError

        .. warning:: `self.fit` has to be called in advance.
        """
        if self.evects is None:
            raise TypeError('The eigenvectors have not been computed.'
                             'You have to perform the fit method.')
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
            ax.axis([0, n_pars + 1, -1 - 0.1, 1 + 0.1])

        axes.flat[-1].set_xlabel('Eigenvector components')
        fig.tight_layout()
        # tight_layout does not consider suptitle so we adjust it manually
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

        :param numpy.ndarray inputs: array n_samples-by-n_params containing the
            points in the full input space.
        :param numpy.ndarray outputs: array n_samples-by-1 containing the
            corresponding function evaluations.
        :param str filename: if specified, the plot is saved at `filename`.
        :param tuple(int,int) figsize: tuple in inches defining the figure
            size. Defaults to (10, 8).
        :param str title: title of the plot.
        :raises: ValueError, TypeError

        .. warning:: `self.fit` has to be called in advance.

            Plot only available for partitions up to dimension 2.
        """
        if self.evects is None:
            raise TypeError('The eigenvectors have not been computed.'
                             'You have to perform the fit method.')

        plt.figure(figsize=figsize)
        plt.title(title)

        if self.dim == 1:
            plt.scatter(self.transform(inputs)[0],
                        outputs,
                        c='blue',
                        s=40,
                        alpha=0.9,
                        edgecolors='k')
            plt.xlabel('Active variable ' + r'$W_1^T \mathbf{\mu}}$',
                       fontsize=18)
            plt.ylabel(r'$f \, (\mathbf{\mu})$', fontsize=18)
        elif self.dim == 2:
            x = self.transform(inputs)[0]
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
                'Sufficient summary plots cannot be made in more than 2 '
                'dimensions.')

        plt.grid(linestyle='dotted')
        plt.tight_layout()

        if filename:
            plt.savefig(filename)
        else:
            plt.show()
