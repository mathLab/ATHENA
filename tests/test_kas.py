from unittest import TestCase
from contextlib import contextmanager
import numpy as np
import matplotlib.pyplot as plt
from athena.kas import KernelActiveSubspaces
from athena import FeatureMap


@contextmanager
def assert_plot_figures_added():
    """
    Assert that the number of figures is higher than when you started the test
    """
    num_figures_before = plt.gcf().number
    yield
    num_figures_after = plt.gcf().number
    assert num_figures_before < num_figures_after


class TestUtils(TestCase):
    def test_init_W1(self):
        ss = KernelActiveSubspaces(dim=2)
        self.assertIsNone(ss.W1)

    def test_init_W2(self):
        ss = KernelActiveSubspaces(dim=2)
        self.assertIsNone(ss.W2)

    def test_init_evals(self):
        ss = KernelActiveSubspaces(dim=2)
        self.assertIsNone(ss.evals)

    def test_init_evects(self):
        ss = KernelActiveSubspaces(dim=2)
        self.assertIsNone(ss.evects)

    def test_init_evals_br(self):
        ss = KernelActiveSubspaces(dim=2)
        self.assertIsNone(ss.evals_br)

    def test_init_subs_br(self):
        ss = KernelActiveSubspaces(dim=2)
        self.assertIsNone(ss.subs_br)

    def test_init_dim(self):
        ss = KernelActiveSubspaces(dim=2)
        self.assertEqual(ss.dim, 2)

    def test_init_n_features(self):
        ss = KernelActiveSubspaces(dim=2)
        self.assertIsNone(ss.n_features)

    def test_init_feature_map(self):
        ss = KernelActiveSubspaces(dim=2)
        self.assertIsNone(ss.feature_map)

    def test_init_features(self):
        ss = KernelActiveSubspaces(dim=2)
        self.assertIsNone(ss.features)

    def test_init_pseudo_gradients(self):
        ss = KernelActiveSubspaces(dim=2)
        self.assertIsNone(ss.pseudo_gradients)

    def test_reparametrize_01(self):
        np.random.seed(42)
        gradients = np.random.uniform(-1, 1, 6).reshape(3, 1, 2)
        inputs = np.random.uniform(-1, 1, 6).reshape(3, 2)
        fm = FeatureMap(distr='multivariate_normal',
                        bias=np.ones((1, 2)),
                        input_dim=inputs.shape[1],
                        n_features=2,
                        params=np.ones(inputs.shape[1]),
                        sigma_f=1)
        ss = KernelActiveSubspaces(dim=2, feature_map=fm)
        pseudo_gradients = ss._reparametrize(inputs, gradients)[0]
        true_pseudo_gradients = np.array([[[-1.17123517, 0.69671587]],
                                          [[0.29308353, 1.12067323]],
                                          [[-0.0036059, -1.75277398]]])
        np.testing.assert_array_almost_equal(pseudo_gradients,
                                             true_pseudo_gradients)

    def test_reparametrize_02(self):
        np.random.seed(42)
        gradients = np.random.uniform(-1, 1, 6).reshape(3, 1, 2)
        inputs = np.random.uniform(-1, 1, 6).reshape(3, 2)
        fm = FeatureMap(distr='multivariate_normal',
                        bias=np.ones((1, 2)),
                        input_dim=inputs.shape[1],
                        n_features=2,
                        params=np.ones(inputs.shape[1]),
                        sigma_f=1)
        ss = KernelActiveSubspaces(dim=2, feature_map=fm)
        features = ss._reparametrize(inputs, gradients)[1]
        true_features = np.array([[-0.2391454, 0.48143467],
                                  [0.42589822, 0.75674833],
                                  [-0.37950285, 0.53470539]])
        np.testing.assert_array_almost_equal(features, true_features)

    def test_compute_01(self):
        ss = KernelActiveSubspaces(dim=2)
        with self.assertRaises(TypeError):
            ss.fit()

    def test_compute_02(self):
        np.random.seed(42)
        gradients = np.random.uniform(-1, 1, 30).reshape(15, 2)
        inputs = np.random.uniform(-1, 1, 30).reshape(15, 2)
        weights = np.ones((15, 1)) / 15
        ss = KernelActiveSubspaces(dim=2,
                                   n_features=4,
                                   method='exact',
                                   n_boot=49)
        ss.fit(inputs=inputs, gradients=gradients, weights=weights)
        true_evals = np.array([0.42588097, 0.19198234, 0.08228976, 0.0068496])
        np.testing.assert_array_almost_equal(true_evals, ss.evals)

    def test_compute_03(self):
        np.random.seed(42)
        gradients = np.random.uniform(-1, 1, 30).reshape(15, 2)
        inputs = np.random.uniform(-1, 1, 30).reshape(15, 2)
        weights = np.ones((15, 1)) / 15
        ss = KernelActiveSubspaces(dim=2,
                                   n_features=4,
                                   method='exact',
                                   n_boot=49)
        ss.fit(inputs=inputs, gradients=gradients, weights=weights)
        true_evects = np.array(
            [[0.74714817, 0.6155644, 0.23414206, 0.08959675],
             [0.35380297, -0.10917583, -0.91115623, 0.18082704],
             [-0.50287165, 0.76801638, -0.33072226, -0.21884635],
             [-0.25241469, 0.1389674, 0.07479708, 0.95466239]])
        np.testing.assert_array_almost_equal(true_evects, ss.evects)

    def test_compute_05(self):
        np.random.seed(42)
        inputs = np.random.uniform(-1, 1, 30).reshape(15, 2)
        outputs = np.random.uniform(0, 5, 15).reshape(15, 1)
        weights = np.ones((15, 1)) / 15
        ss = KernelActiveSubspaces(dim=2,
                                   feature_map=None,
                                   n_features=4,
                                   method='local',
                                   n_boot=49)
        ss.fit(inputs=inputs, outputs=outputs, weights=weights)
        true_evals = np.array(
            [173.56222204, 96.19314922, 29.05560411, 0.85385631])
        np.testing.assert_array_almost_equal(true_evals, ss.evals)

    def test_compute_06(self):
        np.random.seed(42)
        inputs = np.random.uniform(-1, 1, 30).reshape(15, 2)
        outputs = np.random.uniform(0, 5, 15).reshape(15, 1)
        weights = np.ones((15, 1)) / 15
        ss = KernelActiveSubspaces(dim=2,
                                   n_features=4,
                                   method='local',
                                   n_boot=49)
        ss.fit(inputs=inputs, outputs=outputs, weights=weights)
        true_evects = np.array(
            [[0.27316542, 0.65012729, 0.24857554, 0.66402211],
             [-0.34261047, 0.46028689, 0.61561027, -0.54016483],
             [-0.68249783, -0.37635433, 0.35472274, 0.51645514],
             [0.58497472, -0.47310455, 0.65833576, -0.02388905]])
        np.testing.assert_array_almost_equal(true_evects, ss.evects)

    def test_compute_07(self):
        np.random.seed(42)
        inputs = np.random.uniform(-1, 1, 60).reshape(15, 4)
        gradients = np.random.uniform(-1, 1, 180).reshape(15, 3, 4)
        weights = np.ones((15, 1)) / 15
        ss = KernelActiveSubspaces(dim=2, n_features=4, n_boot=49)
        ss.fit(inputs=inputs, gradients=gradients, weights=weights)
        true_evals = np.array(
            [874.84255146, 62.83226559, 3.60417077, 2.84686573])
        np.testing.assert_array_almost_equal(true_evals, ss.evals)

    def test_compute_08(self):
        np.random.seed(42)
        inputs = np.random.uniform(-1, 1, 60).reshape(15, 4)
        gradients = np.random.uniform(-1, 1, 180).reshape(15, 3, 4)
        weights = np.ones((15, 1)) / 15
        ss = KernelActiveSubspaces(dim=2, n_features=4, n_boot=49)
        ss.fit(inputs=inputs, gradients=gradients, weights=weights)
        true_evects = np.array(
            [[0.00126244, 0.99791389, 0.02926469, 0.05753138],
             [0.04385229, -0.05833941, 0.78953331, 0.60935265],
             [-0.99902507, -0.001436, 0.03167332, 0.03071887],
             [0.00492877, -0.02761026, -0.61219077, 0.79021253]])
        np.testing.assert_array_almost_equal(true_evects, ss.evects)

    def test_compute_09(self):
        np.random.seed(42)
        inputs = np.random.uniform(-1, 1, 60).reshape(15, 4)
        gradients = np.random.uniform(-1, 1, 180).reshape(15, 3, 4)
        weights = np.ones((15, 1)) / 15
        metric = np.diag(2 * np.ones(3))
        ss = KernelActiveSubspaces(dim=2,
                                   n_features=4,
                                   method='exact',
                                   n_boot=49)
        ss.fit(inputs=inputs,
               gradients=gradients,
               weights=weights,
               metric=metric)
        true_evects = np.array(
            [[0.00126244, 0.99791389, 0.02926469, 0.05753138],
             [0.04385229, -0.05833941, 0.78953331, 0.60935265],
             [-0.99902507, -0.001436, 0.03167332, 0.03071887],
             [0.00492877, -0.02761026, -0.61219077, 0.79021253]])
        np.testing.assert_array_almost_equal(true_evects, ss.evects)

    def test_compute_10(self):
        np.random.seed(42)
        inputs = np.random.uniform(-1, 1, 60).reshape(15, 4)
        outputs = np.random.uniform(0, 5, 45).reshape(15, 3)
        weights = np.ones((15, 1)) / 15
        ss = KernelActiveSubspaces(dim=2,
                                   n_features=4,
                                   method='local',
                                   n_boot=49)
        ss.fit(inputs=inputs, outputs=outputs, weights=weights)
        true_evals = np.array(
            [7.93870724e+04, 1.18699831e+02, 4.36634158e+01, 1.49812189e+01])
        np.testing.assert_allclose(true_evals, ss.evals)

    def test_compute_11(self):
        np.random.seed(42)
        inputs = np.random.uniform(-1, 1, 60).reshape(15, 4)
        outputs = np.random.uniform(0, 5, 45).reshape(15, 3)
        weights = np.ones((15, 1)) / 15
        metric = np.diag(2 * np.ones(3))
        ss = KernelActiveSubspaces(dim=2,
                                   n_features=4,
                                   method='local',
                                   n_boot=49)
        ss.fit(inputs=inputs, outputs=outputs, weights=weights, metric=metric)
        true_evals = np.array(
            [1.58774145e+05, 2.37399662e+02, 8.73268317e+01, 2.99624379e+01])
        np.testing.assert_allclose(true_evals, ss.evals)

    def test_transform_01(self):
        np.random.seed(42)
        gradients = np.random.uniform(-1, 1, 30).reshape(15, 1, 2)
        inputs = np.random.uniform(-1, 1, 30).reshape(15, 2)
        weights = np.ones((15, 1)) / 15
        ss = KernelActiveSubspaces(dim=2, n_features=4, n_boot=49)
        ss.fit(inputs=inputs, gradients=gradients, weights=weights)
        active = ss.transform(np.random.uniform(-1, 1, 4).reshape(2, 2))[0]
        true_active = np.array([[0.94893046, 0.01774345],
                                [1.09617095, -0.20832091]])
        np.testing.assert_array_almost_equal(true_active, active)

    def test_transform_02(self):
        np.random.seed(42)
        gradients = np.random.uniform(-1, 1, 30).reshape(15, 1, 2)
        inputs = np.random.uniform(-1, 1, 30).reshape(15, 2)
        weights = np.ones((15, 1)) / 15
        ss = KernelActiveSubspaces(dim=2,
                                   n_features=4,
                                   method='exact',
                                   n_boot=49)
        ss.fit(inputs=inputs, gradients=gradients, weights=weights)
        inactive = ss.transform(np.random.uniform(-1, 1, 4).reshape(2, 2))[1]
        true_inactive = np.array([[-0.33551797, 0.36254188],
                                  [-0.19427817, 0.2576207]])
        np.testing.assert_array_almost_equal(true_inactive, inactive)

    def test_transform_03(self):
        np.random.seed(42)
        inputs = np.random.uniform(-1, 1, 60).reshape(15, 4)
        outputs = np.random.uniform(0, 5, 45).reshape(15, 3)
        ss = KernelActiveSubspaces(dim=2, method='local', n_boot=50)
        ss.fit(inputs=inputs, outputs=outputs, metric=np.diag(np.ones(3)))
        active = ss.transform(np.random.uniform(-1, 1, 8).reshape(2, 4))[0]
        true_active = np.array([[-0.18946138, 0.31916713],
                                [-0.25310859, -0.30280365]])
        np.testing.assert_array_almost_equal(true_active, active)

    def test_transform_04(self):
        np.random.seed(42)
        inputs = np.random.uniform(-1, 1, 60).reshape(15, 4)
        outputs = np.random.uniform(0, 5, 15)
        ss = KernelActiveSubspaces(dim=2, method='local', n_boot=49)
        ss.fit(inputs=inputs, outputs=outputs)
        inactive = ss.transform(np.random.uniform(-1, 1, 8).reshape(2, 4))[1]
        true_inactive = np.array([[0.27110018, -0.29359021],
                                  [0.76399199, -0.02233936]])
        np.testing.assert_array_almost_equal(true_inactive, inactive)

    def test_partition_01(self):
        np.random.seed(42)
        matrix = np.random.uniform(-1, 1, 9).reshape(3, 3)
        ss = KernelActiveSubspaces(dim=2)
        ss.evects = matrix
        ss._partition()
        np.testing.assert_array_almost_equal(matrix[:, :2], ss.W1)

    def test_partition_02(self):
        np.random.seed(42)
        matrix = np.random.uniform(-1, 1, 9).reshape(3, 3)
        ss = KernelActiveSubspaces(dim=2)
        ss.evects = matrix
        ss._partition()
        np.testing.assert_array_almost_equal(matrix[:, 2:], ss.W2)

    def test_partition_03(self):
        np.random.seed(42)
        matrix = np.random.uniform(-1, 1, 9).reshape(3, 3)
        ss = KernelActiveSubspaces(dim=2.0)
        ss.evects = matrix
        with self.assertRaises(ValueError):
            ss._partition()

    def test_partition_04(self):
        np.random.seed(42)
        matrix = np.random.uniform(-1, 1, 9).reshape(3, 3)
        ss = KernelActiveSubspaces(dim=0)
        ss.evects = matrix
        with self.assertRaises(ValueError):
            ss._partition()

    def test_partition_05(self):
        np.random.seed(42)
        matrix = np.random.uniform(-1, 1, 9).reshape(3, 3)
        ss = KernelActiveSubspaces(dim=4)
        ss.evects = matrix
        with self.assertRaises(ValueError):
            ss._partition()

    def test_bootstrap_replicate_01(self):
        np.random.seed(42)
        matrix = np.random.uniform(-1, 1, 9).reshape(3, 3)
        weights = np.ones((3, 1)) / 3
        ss = KernelActiveSubspaces(dim=2)
        wei = ss._bootstrap_replicate(matrix, weights)[1]
        np.testing.assert_array_almost_equal(weights, wei)

    def test_bootstrap_replicate_02(self):
        np.random.seed(42)
        matrix = np.random.uniform(-1, 1, 9).reshape(3, 3)
        weights = np.ones((3, 1)) / 3
        ss = KernelActiveSubspaces(dim=2)
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
        ss = KernelActiveSubspaces(dim=2,
                                   n_features=4,
                                   method='exact',
                                   n_boot=49)
        ss.fit(inputs=inputs, gradients=gradients, weights=weights)
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
        ss = KernelActiveSubspaces(dim=2, n_features=4, n_boot=49)
        ss.fit(inputs=inputs, gradients=gradients, weights=weights)
        true_bounds_subspace = np.array([[0.01734317, 0.09791063, 0.19840464],
                                         [0.05112582, 0.43105485, 0.92323839],
                                         [0.05890817, 0.27517302, 0.89262039]])
        np.testing.assert_array_almost_equal(true_bounds_subspace, ss.subs_br)

    def test_compute_bootstrap_ranges_03(self):
        np.random.seed(42)
        gradients = np.random.uniform(-1, 1, 60).reshape(30, 1, 2)
        inputs = np.random.uniform(-1, 1, 60).reshape(30, 2)
        weights = np.ones((30, 1)) / 30
        ss = KernelActiveSubspaces(dim=2,
                                   n_features=4,
                                   method='exact',
                                   n_boot=49)
        ss.fit(inputs=inputs, gradients=gradients, weights=weights)
        true_bounds_evals = np.array([[2.59177494, 7.11443789],
                                      [0.5456548, 1.94294036],
                                      [0.05855044, 0.84178668],
                                      [0.01530059, 0.187785]])
        np.testing.assert_array_almost_equal(true_bounds_evals, ss.evals_br)

    def test_compute_bootstrap_ranges_04(self):
        np.random.seed(42)
        gradients = np.random.uniform(-1, 1, 60).reshape(30, 1, 2)
        inputs = np.random.uniform(-1, 1, 60).reshape(30, 2)
        weights = np.ones((30, 1)) / 30
        ss = KernelActiveSubspaces(dim=2, n_features=4, n_boot=49)
        ss.fit(inputs=inputs, gradients=gradients, weights=weights)
        true_bounds_subspace = np.array([[0.01734317, 0.09791063, 0.19840464],
                                         [0.05112582, 0.43105485, 0.92323839],
                                         [0.05890817, 0.27517302, 0.89262039]])
        np.testing.assert_array_almost_equal(true_bounds_subspace, ss.subs_br)

    def test_plot_eigenvalues_01(self):
        ss = KernelActiveSubspaces(dim=2)
        with self.assertRaises(TypeError):
            ss.plot_eigenvalues(figsize=(7, 7), title='Eigenvalues')

    def test_plot_eigenvalues_02(self):
        np.random.seed(42)
        gradients = np.random.uniform(-1, 1, 200).reshape(50, 1, 4)
        inputs = np.random.uniform(-1, 1, 200).reshape(50, 4)
        weights = np.ones((50, 1)) / 50
        ss = KernelActiveSubspaces(dim=2,
                                   n_features=8,
                                   method='exact',
                                   n_boot=5)
        ss.fit(inputs=inputs, gradients=gradients, weights=weights)
        with assert_plot_figures_added():
            ss.plot_eigenvalues(figsize=(7, 7), title='Eigenvalues')

    def test_plot_eigenvalues_03(self):
        np.random.seed(42)
        gradients = np.random.uniform(-1, 1, 200).reshape(50, 1, 4)
        inputs = np.random.uniform(-1, 1, 200).reshape(50, 4)
        weights = np.ones((50, 1)) / 50
        ss = KernelActiveSubspaces(dim=2, n_features=8, n_boot=5)
        ss.fit(inputs=inputs, gradients=gradients, weights=weights)
        with assert_plot_figures_added():
            ss.plot_eigenvalues(n_evals=3, figsize=(7, 7), title='Eigenvalues')

    def test_plot_eigenvectors_01(self):
        ss = KernelActiveSubspaces(dim=2)
        with self.assertRaises(TypeError):
            ss.plot_eigenvectors(figsize=(7, 7), title='Eigenvalues')

    def test_plot_eigenvectors_02(self):
        np.random.seed(42)
        gradients = np.random.uniform(-1, 1, 200).reshape(50, 1, 4)
        inputs = np.random.uniform(-1, 1, 200).reshape(50, 4)
        weights = np.ones((50, 1)) / 50
        ss = KernelActiveSubspaces(dim=2,
                                   n_features=8,
                                   method='exact',
                                   n_boot=5)
        ss.fit(inputs=inputs, gradients=gradients, weights=weights)
        with assert_plot_figures_added():
            ss.plot_eigenvectors(n_evects=2,
                                 figsize=(7, 7),
                                 title='Eigenvectors')

    def test_plot_eigenvectors_03(self):
        np.random.seed(42)
        gradients = np.random.uniform(-1, 1, 200).reshape(50, 1, 4)
        inputs = np.random.uniform(-1, 1, 200).reshape(50, 4)
        weights = np.ones((50, 1)) / 50
        ss = KernelActiveSubspaces(dim=2,
                                   n_features=5,
                                   method='exact',
                                   n_boot=5)
        ss.fit(inputs=inputs, gradients=gradients, weights=weights)
        with assert_plot_figures_added():
            ss.plot_eigenvectors(n_evects=2,
                                 figsize=(5, 8),
                                 labels=[r'$x$', r'$y$', 'q', r'$r$', r'$z$'])

    def test_plot_sufficient_summary_01(self):
        ss = KernelActiveSubspaces(dim=2)
        with self.assertRaises(TypeError):
            ss.plot_sufficient_summary(10, 10)

    def test_plot_sufficient_summary_02(self):
        np.random.seed(42)
        gradients = np.random.uniform(-1, 1, 200).reshape(50, 1, 4)
        inputs = np.random.uniform(-1, 1, 200).reshape(50, 4)
        weights = np.ones((50, 1)) / 50
        ss = KernelActiveSubspaces(dim=3, n_features=8, n_boot=5)
        ss.fit(inputs=inputs, gradients=gradients, weights=weights)
        with self.assertRaises(ValueError):
            ss.plot_sufficient_summary(10, 10)

    def test_plot_sufficient_summary_03(self):
        np.random.seed(42)
        gradients = np.random.uniform(-1, 1, 200).reshape(50, 1, 4)
        inputs = np.random.uniform(-1, 1, 200).reshape(50, 4)
        weights = np.ones((50, 1)) / 50
        ss = KernelActiveSubspaces(dim=2,
                                   n_features=8,
                                   method='exact',
                                   n_boot=5)
        ss.fit(inputs=inputs, gradients=gradients, weights=weights)
        with assert_plot_figures_added():
            ss.plot_sufficient_summary(
                np.random.uniform(-1, 1, 100).reshape(25, 4),
                np.random.uniform(-1, 1, 25).reshape(-1, 1))
