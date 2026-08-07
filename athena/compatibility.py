"""
Compatibility layer for handling different package versions.
This module provides uniform interfaces for functionality that might
depend on specific versions of packages or alternative implementations.
"""
import numpy as np
import warnings
from packaging import version

# Check if scikit-learn-extra's KMedoids is usable
# with the current NumPy version
SKLEARN_EXTRA_AVAILABLE = False
try:
    import sklearn_extra
    from sklearn_extra.cluster import KMedoids as SklearnExtraKMedoids
    SKLEARN_EXTRA_AVAILABLE = True

    # Check if NumPy version is compatible with sklearn_extra
    if version.parse(np.__version__) >= version.parse('2.0.0'):
        warnings.warn(
            "You are using NumPy >= 2.0.0 with scikit-learn-extra which may "
            "cause compatibility issues. If you encounter errors, consider "
            "using the built-in KMedoids implementation in ATHENA.")
except ImportError:
    SklearnExtraKMedoids = None


# Implementation based on scikit-learn's KMeans but adapted for KMedoids
class KMedoids:
    """
    K-Medoids clustering.
    
    A custom implementation that doesn't rely on scikit-learn-extra, thus
    ensuring compatibility with NumPy 2.0+.
    
    Parameters
    ----------
    n_clusters : int, default=8
        The number of clusters to form as well as the number of medoids to generate.
    
    init : {'k-medoids++', 'random'} or array of shape (n_clusters, n_features), default='k-medoids++'
        Method for initialization.
    
    max_iter : int, default=300
        Maximum number of iterations of the k-medoids algorithm for a single run.
    
    random_state : int, RandomState instance or None, default=None
        Determines random number generation for centroid initialization.
    """

    def __init__(self,
                 n_clusters=8,
                 init='k-medoids++',
                 max_iter=300,
                 random_state=None):
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = 0

    def _init_medoids(self, X):
        """Initialize the medoids."""
        rng = np.random.RandomState(self.random_state)
        n_samples = X.shape[0]

        if isinstance(self.init, str) and self.init == 'random':
            # Random selection
            indices = rng.permutation(n_samples)[:self.n_clusters]
            self.cluster_centers_ = X[indices].copy()
        elif isinstance(self.init, str) and self.init == 'k-medoids++':
            # Implementation of k-medoids++ initialization
            # Choose the first medoid randomly
            indices = np.zeros(self.n_clusters, dtype=int)
            indices[0] = rng.randint(n_samples)

            # Calculate distances to the first medoid
            distances = np.sum((X - X[indices[0]])**2, axis=1)

            # Choose remaining medoids
            for i in range(1, self.n_clusters):
                # Choose point with probability proportional to distance squared
                probs = distances / np.sum(distances)
                indices[i] = rng.choice(n_samples, p=probs)

                # Update distances
                new_dist = np.sum((X - X[indices[i]])**2, axis=1)
                distances = np.minimum(distances, new_dist)

            self.cluster_centers_ = X[indices].copy()
        else:
            # Use provided initial medoids
            self.cluster_centers_ = np.asarray(self.init, dtype=X.dtype)

    def fit(self, X):
        """Compute k-medoids clustering."""
        X = np.asarray(X)
        self._init_medoids(X)

        best_labels = None
        best_inertia = float('inf')
        best_centers = None

        for i in range(self.max_iter):
            # Assign each point to closest medoid
            distances = np.zeros((X.shape[0], self.n_clusters))
            for j in range(self.n_clusters):
                distances[:, j] = np.sum((X - self.cluster_centers_[j])**2,
                                         axis=1)

            labels = np.argmin(distances, axis=1)

            # Update medoids
            old_centers = self.cluster_centers_.copy()

            # For each cluster, update medoid to be the point minimizing inertia
            for j in range(self.n_clusters):
                cluster_points = X[labels == j]
                if len(cluster_points) > 0:
                    # Compute pairwise distances within cluster
                    inertias = np.zeros(len(cluster_points))
                    for k, point in enumerate(cluster_points):
                        inertias[k] = np.sum(
                            np.sum((cluster_points - point)**2, axis=1))

                    # Choose point with minimal inertia as new medoid
                    min_idx = np.argmin(inertias)
                    self.cluster_centers_[j] = cluster_points[min_idx].copy()

            # Compute inertia
            inertia = 0
            for j in range(self.n_clusters):
                cluster_points = X[labels == j]
                if len(cluster_points) > 0:
                    inertia += np.sum(
                        np.sum((cluster_points - self.cluster_centers_[j])**2,
                               axis=1))

            # Store best result
            if inertia < best_inertia:
                best_inertia = inertia
                best_labels = labels
                best_centers = self.cluster_centers_.copy()

            # Check for convergence
            center_shift = np.sum(
                np.sqrt(np.sum((old_centers - self.cluster_centers_)**2,
                               axis=1)))
            if center_shift < 1e-4:
                break

        self.labels_ = best_labels
        self.cluster_centers_ = best_centers
        self.inertia_ = best_inertia
        self.n_iter_ = i + 1

        return self

    def predict(self, X):
        """Predict the closest cluster for each sample in X."""
        X = np.asarray(X)
        distances = np.zeros((X.shape[0], self.n_clusters))
        for j in range(self.n_clusters):
            distances[:, j] = np.sum((X - self.cluster_centers_[j])**2, axis=1)

        return np.argmin(distances, axis=1)


# Export the appropriate KMedoids implementation
if SKLEARN_EXTRA_AVAILABLE and version.parse(
        np.__version__) < version.parse('2.0.0'):
    # Use sklearn-extra's implementation when available and NumPy < 2.0
    KMedoids = SklearnExtraKMedoids
# Otherwise use our implementation which is compatible with NumPy 2.0+
