"""
Module classification with local Active Subspaces dimension.

:References:

    - Romor Francesco, Marco Tezzele, and Gianluigi Rozza.
      "A local approach to parameter space reduction for regression and classification tasks." arXiv preprint arXiv:2107.10867 (2021).

"""

import abc
import itertools
import logging

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

import numpy as np

_log = logging.getLogger('classify_as')
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)

plt.rcParams.update({'font.size': 14})


class SpectralClassification(metaclass=abc.ABCMeta):
    """Evaluate the connected components from X, n_neighbours, features and custom
    distance that must be defined in concrete class."""
    def __init__(self):
        self.X = None
        self.features = None
        self.labels = None
        self.n_neighbours = None
        self.n_components = None
        self.graph = None

    @abc.abstractstaticmethod
    def custom_distance(X: np.ndarray) -> np.ndarray:
        return NotImplementedError

    @abc.abstractmethod
    def _init_values(self, *args, **kwargs):
        return NotImplementedError

    def fit(self, *args, **kwargs):
        self._init_values(*args, **kwargs)
        assert self.n_neighbours is not None and self.features is not None, "Features and n_neighboours variables not initialized."

        self.graph = self.make_graph()
        n_components, labels = connected_components(csgraph=self.graph,
                                                    directed=True,
                                                    return_labels=True)
        self.n_components = n_components
        self.labels = labels
        return n_components, labels

    def plot(self, id1=0, id2=1, save=False):
        """Plot coordinates id1, id2 with local clustering based on as_dimension
        and local as dimension information."""
        assert (self.n_components is not None)
        assert (self.labels is not None)

        fig = plt.figure(figsize=(16, 6))
        ax1 = fig.add_subplot(121)
        colors = cm.rainbow(np.linspace(0, 1, self.n_components))
        for i in range(self.n_components):
            # dim_mask = self.features[:, -1] == i + 1
            dim_mask = self.labels == i
            ax1.scatter(
                self.X[dim_mask, id1],
                self.X[dim_mask, id2],
                c=colors[i].reshape(1, -1),
                label=f"id {str(i)}",
            )
        plt.title("clusters")
        plt.grid()
        plt.legend()

        ax1 = fig.add_subplot(122)
        colors1 = cm.rainbow(np.linspace(0, 1, self.X.shape[1]))
        for i in range(self.X.shape[1]):
            dim_mask = self.features[:, -1] == i + 1
            ax1.scatter(
                self.X[dim_mask, id1],
                self.X[dim_mask, id2],
                c=colors1[i].reshape(1, -1),
                label=f"dim {str(i + 1)}",
            )
        plt.title("as dimensions")
        plt.grid()
        plt.legend()

        if save:
            save_dat = np.hstack(
                (self.X, self.labels.reshape(-1, 1), self.features[:, -1:]))
            print("Saved dat with shape: ", save_dat.shape)
            np.save("clusters_labels_localasdim.npy", save_dat)

    def make_graph(self):
        """Use scipy sparse format COO to create adjacency list"""
        distance_matrix = self.custom_distance(self.features)
        _log.debug(f"distance matrix {distance_matrix}")

        adjacency_list = np.argsort(distance_matrix,
                                    axis=1)[:, :self.n_neighbours]
        _log.debug(f"adjacency list {adjacency_list}")

        distance_restricted = np.take_along_axis(distance_matrix,
                                                 adjacency_list,
                                                 axis=1)
        inf_mask = np.isinf(distance_restricted)
        # evaluate the number of neighbours with non-inf distance subtracting
        # the array of non-inf distance neighbours. Broadcasting is used
        neighbours_count = np.array([self.n_neighbours]) - np.count_nonzero(
            inf_mask, axis=1)

        _log.debug(f"neighbours count {neighbours_count}")
        assert (all(neighbours_count >= 1))

        row = np.hstack((i * np.ones(neighbours_count[i], dtype=np.int8)
                         for i in range(neighbours_count.shape[0])))
        _log.debug(f"row {row}")

        col = np.hstack((adjacency_list[i, :neighbours_count[i]]
                         for i in range(neighbours_count.shape[0])))
        _log.debug(f"col {col}")

        data = np.ones(np.sum(neighbours_count), dtype=np.int8)
        _log.debug(f"data {data}")

        assert (data.shape[0] == row.shape[0] == col.shape[0])
        return coo_matrix((data, (row, col)), dtype=np.int8)

    def plot_decision_boundaries(self, true_inputs=None, true_labels=None):
        """Use scikit-learn classification algorithms to plot decision
        boundaries"""
        assert self.n_components is not None
        assert self.labels is not None
        if true_inputs is not None and true_labels is not None:
            return decision_boundaries(self.features[:, :self.X.shape[1]],
                                       self.labels.reshape(-1, 1), true_inputs,
                                       true_labels, self.n_components)
        else:
            return decision_boundaries(self.features[:, :self.X.shape[1]],
                                       self.labels.reshape(-1, 1),
                                       self.n_components)


class ClassifyAS(SpectralClassification):
    """Compute the local AS dimension of every input-gradient sample, evaluating
    the AS dimension of the n_neighbours neighbouring samples with a resampling
    of neighbour_resampling. The local_as_criterion can be 'min' or 'average'
    over the batches of neighbouring samples."""
    def __init__(self):
        super().__init__()

        self.X = None
        self.features = None
        self.n_neighbours = None
        self.threshold = None
        self.neighbour_resampling = None
        self.local_as_criterion = None

    def _init_values(self,
                     inputs,
                     gradients,
                     n_neighbours=None,
                     threshold=None,
                     neighbour_resampling=5,
                     local_as_criterion='min'):
        if n_neighbours is None:
            n_neighbours = inputs.shape[1]
        if threshold is None:
            threshold = 0.999999

        assert (0 < threshold <= 1)
        assert (n_neighbours >= neighbour_resampling)

        self.X = inputs
        self.n_neighbours = n_neighbours
        self.threshold = threshold
        self.neighbour_resampling = neighbour_resampling
        self.local_as_criterion = local_as_criterion

        self.features = np.hstack(
            (inputs, self.evaluate_minimum_distance(inputs, gradients)))

    @staticmethod
    def custom_distance(X):
        n_samples = X.shape[0]
        distance_matrix = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                distance_matrix[i, j] = ClassifyAS.distance(X[i], X[j])
        return distance_matrix + distance_matrix.T

    @staticmethod
    def distance(x, y):
        # if two nodes have different AS dimension the distance is inf
        if np.abs(x[-1] - y[-1]) != 0:
            return np.inf
            # when two nodes have the same AS dimension the distance is the euclidean one
        else:
            return np.linalg.norm(x[:-1] - y[:-1])

    def evaluate_minimum_distance(self, X, dX):
        """Evaluate local active subspace dimension depending on the choice of
        the number of neighbouring points and of subsampling from the same
        neighbour in order to evaluate the local active subspace"""

        assert (X.shape == dX.shape)

        as_dim_features = np.zeros(X.shape[0])

        for i in range(X.shape[0]):
            D = np.linalg.norm((X - X[i].reshape(1, -1)), axis=1)
            ind = np.argsort(D)
            _log.debug(f"ind {i}, neigh {ind}, dist {D}")

            res = []

            for mask in itertools.combinations(np.arange(self.n_neighbours),
                                               self.neighbour_resampling):
                singular = np.linalg.svd(dX[ind[:self.n_neighbours]][mask, :],
                                         full_matrices=False)[1]
                evals = singular**2
                if evals[0] > 0: evals /= np.sum(evals)

                as_dim = 0
                for cumulative in itertools.accumulate(evals):
                    as_dim += 1
                    if cumulative > self.threshold: break
                res.append(as_dim)

                _log.debug(
                    f" {mask} evals {as_dim}, res {res}, {evals[:4]}, \n{dX[ind[:self.n_neighbours]][mask, :]}"
                )

            _log.debug(
                f"{i} as dimensiones {int(as_dim_features[i] / self.n_neighbours)} approx {as_dim_features[i] / self.n_neighbours}"
            )

            if self.local_as_criterion == 'min':
                as_dim_features[i] = round(min(res))
            elif self.local_as_criterion == 'mean':
                as_dim_features[i] = round(sum(res) / self.n_neighbours)
            else:
                raise ValueError(
                    f"The local_as_criterion must be 'min' or 'mean'. Passed value is {self.local_as_criterion}"
                )

        _log.debug(f"as dimensiones {as_dim_features}")
        return as_dim_features.reshape(-1, 1)


def decision_boundaries(X,
                        y,
                        true_inputs=None,
                        true_labels=None,
                        components=(0, 1),
                        plot=True,
                        test_size=0.2,
                        classifier=MLPClassifier(alpha=1, max_iter=1000)):

    _log.debug(f"Shapes: {X.shape} {y.shape} ")

    clf = classifier
    h = .02  # step size in the mesh

    # preprocess dataset, split into training and test part
    # X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=test_size,
                                                        random_state=42)

    _log.debug(
        f"Shapes train, test: {X_train.shape}, {X_test.shape}, {y_train.shape}, {y_test.shape}"
    )

    x_min, x_max = X[:, components[0]].min() - .1, X[:,
                                                     components[0]].max() + .1
    y_min, y_max = X[:, components[1]].min() - .1, X[:,
                                                     components[1]].max() + .1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # train lassifier
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)

    if true_inputs is not None and true_labels is not None:
        score_true = clf.score(true_inputs, true_labels)
        _log.debug(f"Score {score} {score_true}")

    if plot:
        # Plot the decision boundary. For that, we will assign a color to
        # each point in the mesh [x_min, x_max]x[y_min, y_max].
        if hasattr(clf, "decision_function"):
            _log.debug(xx.ravel().shape)
            Z = clf.decision_function(
                np.hstack((np.c_[xx.ravel(), yy.ravel()],
                           np.zeros((xx.ravel().shape[0], X.shape[1] - 2)))))
            _log.debug(f"Decision function  {Z.shape}")
        else:
            Z = clf.predict_proba(
                np.hstack((np.c_[xx.ravel(), yy.ravel()],
                           np.zeros((xx.ravel().shape[0], X.shape[1] - 2)))))
            _log.debug(f"Predictions probability {Z.shape[1]}")

        n_labels = Z.shape[1]

        fig, ax = plt.subplots(1, n_labels, figsize=(4 * n_labels, 5))
        plt.suptitle("Classification of the local AS dimension with \n" +
                     str(clf) +
                     "\nMean accuracy on the test set: {:.2f}".format(score))

        for l in range(n_labels):
            Z_ = Z[:, l].reshape(xx.shape)
            ax[l].contourf(xx, yy, Z_, cmap=cm.RdBu, alpha=.2)

            if true_inputs is not None and true_labels is not None:
                ax[l].scatter(true_inputs[:, components[0]],
                              true_inputs[:, components[1]],
                              c=true_labels,
                              edgecolors='k',
                              alpha=0.8)

            ax[l].set_xlim(xx.min(), xx.max())
            ax[l].set_ylim(yy.min(), yy.max())
            ax[l].set_xticks(())
            ax[l].set_yticks(())
            ax[l].set_title(f"Local cluster {l}")

        plt.tight_layout()
        plt.show()
        plt.close()

    if true_inputs is not None and true_labels is not None:
        return score, score_true
    else:
        return score
