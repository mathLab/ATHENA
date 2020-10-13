from unittest import TestCase
import numpy as np
from athena.kas import KernelActiveSubspaces
from contextlib import contextmanager
import matplotlib.pyplot as plt


@contextmanager
def assert_plot_figures_added():
    """
    Assert that the number of figures is higher than
    when you started the test
    """
    num_figures_before = plt.gcf().number
    yield
    num_figures_after = plt.gcf().number
    assert num_figures_before < num_figures_after


class TestUtils(TestCase):
    def test_init_W1(self):
        ss = KernelActiveSubspaces()
        self.assertIsNone(ss.W1)

    def test_init_W2(self):
        ss = KernelActiveSubspaces()
        self.assertIsNone(ss.W2)

    def test_init_evals(self):
        ss = KernelActiveSubspaces()
        self.assertIsNone(ss.evals)

    def test_init_evects(self):
        ss = KernelActiveSubspaces()
        self.assertIsNone(ss.evects)

    def test_init_evals_br(self):
        ss = KernelActiveSubspaces()
        self.assertIsNone(ss.evals_br)

    def test_init_subs_br(self):
        ss = KernelActiveSubspaces()
        self.assertIsNone(ss.subs_br)

    def test_init_dim(self):
        ss = KernelActiveSubspaces()
        self.assertIsNone(ss.dim)

    def test_init_cov_matrix(self):
        ss = KernelActiveSubspaces()
        self.assertIsNone(ss.cov_matrix)

    def test_init_n_features(self):
        ss = KernelActiveSubspaces()
        self.assertIsNone(ss.n_features)

    def test_init_feature_map(self):
        ss = KernelActiveSubspaces()
        self.assertIsNone(ss.feature_map)

    def test_init_features(self):
        ss = KernelActiveSubspaces()
        self.assertIsNone(ss.features)

    def test_init_pseudo_gradients(self):
        ss = KernelActiveSubspaces()
        self.assertIsNone(ss.pseudo_gradients)

    def test_compute_01(self):
        ss = KernelActiveSubspaces()
        with self.assertRaises(ValueError):
            ss.compute()

    def test_compute_02(self):
        np.random.seed(42)
        gradients = np.random.uniform(-1, 1, 30).reshape(15, 2)
        inputs = np.random.uniform(-1, 1, 30).reshape(15, 2)
        weights = np.ones((15, 1)) / 15
        ss = KernelActiveSubspaces()
        ss.compute(inputs=inputs,
                   gradients=gradients,
                   weights=weights,
                   method='exact',
                   nboot=49,
                   n_features=4,
                   feature_map=None)
        true_evals = np.array([0.42588097, 0.19198234, 0.08228976, 0.0068496])
        np.testing.assert_array_almost_equal(true_evals, ss.evals)

    def test_compute_03(self):
        np.random.seed(42)
        gradients = np.random.uniform(-1, 1, 30).reshape(15, 2)
        inputs = np.random.uniform(-1, 1, 30).reshape(15, 2)
        weights = np.ones((15, 1)) / 15
        ss = KernelActiveSubspaces()
        ss.compute(inputs=inputs,
                   gradients=gradients,
                   weights=weights,
                   method='exact',
                   nboot=49,
                   n_features=4,
                   feature_map=None)
        true_evects = np.array(
            [[0.74714817, 0.6155644, 0.23414206, 0.08959675],
             [0.35380297, -0.10917583, -0.91115623, 0.18082704],
             [-0.50287165, 0.76801638, -0.33072226, -0.21884635],
             [-0.25241469, 0.1389674, 0.07479708, 0.95466239]])
        np.testing.assert_array_almost_equal(true_evects, ss.evects)

    # def test_compute_05(self):
    #     np.random.seed(42)
    #     inputs = np.random.uniform(-1, 1, 30).reshape(15, 2)
    #     outputs = np.random.uniform(0, 5, 15).reshape(15, 1)
    #     weights = np.ones((15, 1)) / 15
    #     ss = KernelActiveSubspaces()
    #     ss.compute(inputs=inputs,
    #                outputs=outputs,
    #                weights=weights,
    #                method='local',
    #                nboot=49,
    #                n_features=4,
    #                feature_map=None)
    #     true_evals = np.array([[13.794711], [11.102377], [3.467318],
    #                            [1.116324]])
    #     np.testing.assert_array_almost_equal(true_evals, ss.evals)

    # def test_compute_06(self):
    #     np.random.seed(42)
    #     inputs = np.random.uniform(-1, 1, 30).reshape(15, 2)
    #     outputs = np.random.uniform(0, 5, 15).reshape(15, 1)
    #     weights = np.ones((15, 1)) / 15
    #     ss = KernelActiveSubspaces()
    #     ss.compute(inputs=inputs,
    #                outputs=outputs,
    #                weights=weights,
    #                method='local',
    #                nboot=49,
    #                n_features=4,
    #                feature_map=None)
    #     true_evects = np.array([[0.164383, 0.717021, 0.237246, 0.634486],
    #                             [0.885808, 0.177628, -0.004112, -0.428691],
    #                             [-0.255722, 0.558199, -0.734083, -0.290071],
    #                             [-0.350612, 0.377813, 0.636254, -0.574029]])
    #     np.testing.assert_array_almost_equal(true_evects, ss.evects)

    def test_forward_01(self):
        np.random.seed(42)
        gradients = np.random.uniform(-1, 1, 30).reshape(15, 1, 2)
        inputs = np.random.uniform(-1, 1, 30).reshape(15, 2)
        weights = np.ones((15, 1)) / 15
        ss = KernelActiveSubspaces()
        ss.compute(inputs=inputs,
                   gradients=gradients,
                   weights=weights,
                   method='exact',
                   nboot=49,
                   n_features=4,
                   feature_map=None)
        ss.partition(2)
        active = ss.forward(np.random.uniform(-1, 1, 4).reshape(2, 2))[0]
        true_active = np.array([[1.34199032, 0.02509303],
                                [1.55021982, -0.29461026]])
        np.testing.assert_array_almost_equal(true_active, active)

    def test_forward_02(self):
        np.random.seed(42)
        gradients = np.random.uniform(-1, 1, 30).reshape(15, 1, 2)
        inputs = np.random.uniform(-1, 1, 30).reshape(15, 2)
        weights = np.ones((15, 1)) / 15
        ss = KernelActiveSubspaces()
        ss.compute(inputs=inputs,
                   gradients=gradients,
                   weights=weights,
                   method='exact',
                   nboot=49,
                   n_features=4,
                   feature_map=None)
        ss.partition(2)
        inactive = ss.forward(np.random.uniform(-1, 1, 4).reshape(2, 2))[1]
        print(inactive)
        true_inactive = np.array([[-0.47449407, 0.51271165],
                                  [-0.27475082, 0.36433068]])
        np.testing.assert_array_almost_equal(true_inactive, inactive)

    def test_partition_01(self):
        np.random.seed(42)
        matrix = np.random.uniform(-1, 1, 9).reshape(3, 3)
        ss = KernelActiveSubspaces()
        ss.evects = matrix
        ss.partition(dim=2)
        np.testing.assert_array_almost_equal(matrix[:, :2], ss.W1)

    def test_partition_02(self):
        np.random.seed(42)
        matrix = np.random.uniform(-1, 1, 9).reshape(3, 3)
        ss = KernelActiveSubspaces()
        ss.evects = matrix
        ss.partition(dim=2)
        np.testing.assert_array_almost_equal(matrix[:, 2:], ss.W2)

    def test_partition_03(self):
        np.random.seed(42)
        matrix = np.random.uniform(-1, 1, 9).reshape(3, 3)
        ss = KernelActiveSubspaces()
        ss.evects = matrix
        with self.assertRaises(TypeError):
            ss.partition(dim=2.0)

    def test_partition_04(self):
        np.random.seed(42)
        matrix = np.random.uniform(-1, 1, 9).reshape(3, 3)
        ss = KernelActiveSubspaces()
        ss.evects = matrix
        with self.assertRaises(ValueError):
            ss.partition(dim=0)

    def test_partition_05(self):
        np.random.seed(42)
        matrix = np.random.uniform(-1, 1, 9).reshape(3, 3)
        ss = KernelActiveSubspaces()
        ss.evects = matrix
        with self.assertRaises(ValueError):
            ss.partition(dim=4)

    def test_bootstrap_replicate_01(self):
        np.random.seed(42)
        matrix = np.random.uniform(-1, 1, 9).reshape(3, 3)
        weights = np.ones((3, 1)) / 3
        ss = KernelActiveSubspaces()
        wei = ss._bootstrap_replicate(matrix, weights)[1]
        np.testing.assert_array_almost_equal(weights, wei)

    def test_bootstrap_replicate_02(self):
        np.random.seed(42)
        matrix = np.random.uniform(-1, 1, 9).reshape(3, 3)
        weights = np.ones((3, 1)) / 3
        ss = KernelActiveSubspaces()
        mat = ss._bootstrap_replicate(matrix, weights)[0]
        true_matrix = np.array([[-0.88383278, 0.73235229, 0.20223002],
                                [0.19731697, -0.68796272, -0.68801096],
                                [-0.25091976, 0.90142861, 0.46398788]])
        np.testing.assert_array_almost_equal(true_matrix, mat)

    def test_compute_bootstrap_ranges_01(self):
        np.random.seed(42)
        gradients = np.random.uniform(-1, 1, 60).reshape(30, 2)
        inputs = np.random.uniform(-1, 1, 60).reshape(30, 2)
        weights = np.ones((30, 1)) / 30
        ss = KernelActiveSubspaces()
        ss.compute(inputs=inputs,
                   gradients=gradients,
                   weights=weights,
                   method='exact',
                   nboot=49,
                   n_features=4,
                   feature_map=None)
        true_bounds_evals = np.array([[2.59177494, 7.11443789],
                                      [0.5456548, 1.94294036],
                                      [0.05855044, 0.84178668],
                                      [0.01530059, 0.187785]])

        np.testing.assert_array_almost_equal(true_bounds_evals, ss.evals_br)

    def test_compute_bootstrap_ranges_02(self):
        np.random.seed(42)
        gradients = np.random.uniform(-1, 1, 60).reshape(30, 1, 2)
        inputs = np.random.uniform(-1, 1, 60).reshape(30, 2)
        weights = np.ones((30, 1)) / 30
        ss = KernelActiveSubspaces()
        ss.compute(inputs=inputs,
                   gradients=gradients,
                   weights=weights,
                   method='exact',
                   nboot=49,
                   n_features=4,
                   feature_map=None)
        true_bounds_subspace = np.array([[0.01734317, 0.09791063, 0.19840464],
                                         [0.05112582, 0.43105485, 0.92323839],
                                         [0.05890817, 0.27517302, 0.89262039]])
        np.testing.assert_array_almost_equal(true_bounds_subspace, ss.subs_br)

    def test_plot_eigenvalues_01(self):
        ss = KernelActiveSubspaces()
        with self.assertRaises(ValueError):
            ss.plot_eigenvalues(figsize=(7, 7), title='Eigenvalues')

    def test_plot_eigenvalues_02(self):
        np.random.seed(42)
        gradients = np.random.uniform(-1, 1, 200).reshape(50, 1, 4)
        inputs = np.random.uniform(-1, 1, 200).reshape(50, 4)
        weights = np.ones((50, 1)) / 50
        ss = KernelActiveSubspaces()
        ss.compute(inputs=inputs,
                   gradients=gradients,
                   weights=weights,
                   method='exact',
                   nboot=49,
                   n_features=8,
                   feature_map=None)
        with assert_plot_figures_added():
            ss.plot_eigenvalues(figsize=(7, 7), title='Eigenvalues')

    def test_plot_eigenvalues_03(self):
        np.random.seed(42)
        gradients = np.random.uniform(-1, 1, 200).reshape(50, 1, 4)
        inputs = np.random.uniform(-1, 1, 200).reshape(50, 4)
        weights = np.ones((50, 1)) / 50
        ss = KernelActiveSubspaces()
        ss.compute(inputs=inputs,
                   gradients=gradients,
                   weights=weights,
                   method='exact',
                   nboot=49,
                   n_features=8,
                   feature_map=None)
        with assert_plot_figures_added():
            ss.plot_eigenvalues(n_evals=3, figsize=(7, 7), title='Eigenvalues')

    def test_plot_eigenvectors_01(self):
        ss = KernelActiveSubspaces()
        with self.assertRaises(ValueError):
            ss.plot_eigenvectors(figsize=(7, 7), title='Eigenvalues')

    def test_plot_eigenvectors_02(self):
        np.random.seed(42)
        gradients = np.random.uniform(-1, 1, 200).reshape(50, 1, 4)
        inputs = np.random.uniform(-1, 1, 200).reshape(50, 4)
        weights = np.ones((50, 1)) / 50
        ss = KernelActiveSubspaces()
        ss.compute(inputs=inputs,
                   gradients=gradients,
                   weights=weights,
                   method='exact',
                   nboot=49,
                   n_features=8,
                   feature_map=None)
        with assert_plot_figures_added():
            ss.plot_eigenvectors(n_evects=2,
                                 figsize=(7, 7),
                                 title='Eigenvectors')

    def test_plot_eigenvectors_03(self):
        np.random.seed(42)
        gradients = np.random.uniform(-1, 1, 200).reshape(50, 1, 4)
        inputs = np.random.uniform(-1, 1, 200).reshape(50, 4)
        weights = np.ones((50, 1)) / 50
        ss = KernelActiveSubspaces()
        ss.compute(inputs=inputs,
                   gradients=gradients,
                   weights=weights,
                   method='exact',
                   nboot=49,
                   n_features=5,
                   feature_map=None)
        with assert_plot_figures_added():
            ss.plot_eigenvectors(n_evects=2,
                                 figsize=(5, 8),
                                 labels=[r'$x$', r'$y$', 'q', r'$r$', r'$z$'])

    def test_plot_sufficient_summary_01(self):
        ss = KernelActiveSubspaces()
        with self.assertRaises(ValueError):
            ss.plot_sufficient_summary(10, 10)

    def test_plot_sufficient_summary_02(self):
        np.random.seed(42)
        gradients = np.random.uniform(-1, 1, 200).reshape(50, 1, 4)
        inputs = np.random.uniform(-1, 1, 200).reshape(50, 4)
        weights = np.ones((50, 1)) / 50
        ss = KernelActiveSubspaces()
        ss.compute(inputs=inputs,
                   gradients=gradients,
                   weights=weights,
                   method='exact',
                   nboot=49,
                   n_features=8,
                   feature_map=None)
        ss.partition(3)
        with self.assertRaises(ValueError):
            ss.plot_sufficient_summary(10, 10)

    def test_plot_sufficient_summary_03(self):
        np.random.seed(42)
        gradients = np.random.uniform(-1, 1, 200).reshape(50, 1, 4)
        inputs = np.random.uniform(-1, 1, 200).reshape(50, 4)
        weights = np.ones((50, 1)) / 50
        ss = KernelActiveSubspaces()
        ss.compute(inputs=inputs,
                   gradients=gradients,
                   weights=weights,
                   method='exact',
                   nboot=49,
                   n_features=8,
                   feature_map=None)
        ss.partition(2)
        with assert_plot_figures_added():
            ss.plot_sufficient_summary(
                np.random.uniform(-1, 1, 100).reshape(25, 4),
                np.random.uniform(-1, 1, 25).reshape(-1, 1))
