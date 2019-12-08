from unittest import TestCase
from athena.active import ActiveSubspaces
import numpy as np
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
        ss = ActiveSubspaces()
        self.assertIsNone(ss.W1)

    def test_init_W2(self):
        ss = ActiveSubspaces()
        self.assertIsNone(ss.W2)

    def test_init_evals(self):
        ss = ActiveSubspaces()
        self.assertIsNone(ss.evals)

    def test_init_evects(self):
        ss = ActiveSubspaces()
        self.assertIsNone(ss.evects)

    def test_init_evals_br(self):
        ss = ActiveSubspaces()
        self.assertIsNone(ss.evals_br)

    def test_init_subs_br(self):
        ss = ActiveSubspaces()
        self.assertIsNone(ss.subs_br)

    def test_init_dim(self):
        ss = ActiveSubspaces()
        self.assertIsNone(ss.dim)

    def test_init_cov_matrix(self):
        ss = ActiveSubspaces()
        self.assertIsNone(ss.cov_matrix)

    def test_compute_01(self):
        ss = ActiveSubspaces()
        with self.assertRaises(ValueError):
            ss.compute()

    def test_compute_02(self):
        np.random.seed(42)
        gradients = np.random.uniform(-1, 1, 60).reshape(15, 4)
        weights = np.ones((15, 1)) / 15
        ss = ActiveSubspaces()
        ss.compute(gradients=gradients,
                   weights=weights,
                   method='exact',
                   nboot=150)
        true_evals = np.array([[0.571596], [0.465819], [0.272198], [0.175012]])
        np.testing.assert_array_almost_equal(true_evals, ss.evals)

    def test_compute_03(self):
        np.random.seed(42)
        gradients = np.random.uniform(-1, 1, 60).reshape(15, 4)
        weights = np.ones((15, 1)) / 15
        ss = ActiveSubspaces()
        ss.compute(gradients=gradients,
                   weights=weights,
                   method='exact',
                   nboot=200)
        true_evects = np.array([[0.019091, 0.408566, 0.861223, 0.301669],
                                [0.767799, -0.199069, 0.268823, -0.546434],
                                [0.463451, 0.758442, -0.427696, 0.164486],
                                [0.441965, -0.467131, -0.055723, 0.763774]])
        np.testing.assert_array_almost_equal(true_evects, ss.evects)

    def test_compute_04(self):
        np.random.seed(42)
        gradients = np.random.uniform(-1, 1, 60).reshape(15, 4)
        weights = np.ones((15, 1)) / 15
        ss = ActiveSubspaces()
        ss.compute(gradients=gradients,
                   weights=weights,
                   method='exact',
                   nboot=250)
        true_cov_matrix = np.array([[0.295783, 0.004661, 0.057825, -0.056819],
                                    [0.004661, 0.427352, 0.086039, 0.160164],
                                    [0.057825, 0.086039, 0.445253, -0.019482],
                                    [-0.056819, 0.160164, -0.019482, 0.316237]])
        np.testing.assert_array_almost_equal(true_cov_matrix, ss.cov_matrix)

    def test_compute_05(self):
        np.random.seed(42)
        inputs = np.random.uniform(-1, 1, 60).reshape(15, 4)
        outputs = np.random.uniform(0, 5, 15)
        ss = ActiveSubspaces()
        ss.compute(inputs=inputs, outputs=outputs, method='local', nboot=150)
        true_evals = np.array([[13.794711], [11.102377], [3.467318],
                               [1.116324]])
        np.testing.assert_array_almost_equal(true_evals, ss.evals)

    def test_compute_06(self):
        np.random.seed(42)
        inputs = np.random.uniform(-1, 1, 60).reshape(15, 4)
        outputs = np.random.uniform(0, 5, 15)
        ss = ActiveSubspaces()
        ss.compute(inputs=inputs, outputs=outputs, method='local', nboot=200)
        true_evects = np.array([[0.164383, 0.717021, 0.237246, 0.634486],
                                [0.885808, 0.177628, -0.004112, -0.428691],
                                [-0.255722, 0.558199, -0.734083, -0.290071],
                                [-0.350612, 0.377813, 0.636254, -0.574029]])
        np.testing.assert_array_almost_equal(true_evects, ss.evects)

    def test_compute_07(self):
        np.random.seed(42)
        inputs = np.random.uniform(-1, 1, 60).reshape(15, 4)
        outputs = np.random.uniform(0, 5, 15)
        ss = ActiveSubspaces()
        ss.compute(inputs=inputs, outputs=outputs, method='local', nboot=250)
        true_cov_matrix = np.array([[6.725267, 3.115678, 3.054427, 2.329383],
                                    [3.115678, 11.379603, -1.87469, -3.273579],
                                    [3.054427, -1.87469, 6.323835, 2.144681],
                                    [2.329383, -3.273579, 2.144681, 5.052025]])
        np.testing.assert_array_almost_equal(true_cov_matrix, ss.cov_matrix)

    def test_forward_01(self):
        np.random.seed(42)
        inputs = np.random.uniform(-1, 1, 60).reshape(15, 4)
        outputs = np.random.uniform(0, 5, 15)
        ss = ActiveSubspaces()
        ss.compute(inputs=inputs, outputs=outputs, method='local', nboot=250)
        ss.partition(2)
        active = ss.forward(np.random.uniform(-1, 1, 8).reshape(2, 4))[0]
        true_active = np.array([[0.004748, 0.331107], [0.949099, 0.347534]])
        np.testing.assert_array_almost_equal(true_active, active)

    def test_forward_02(self):
        np.random.seed(42)
        inputs = np.random.uniform(-1, 1, 60).reshape(15, 4)
        outputs = np.random.uniform(0, 5, 15)
        ss = ActiveSubspaces()
        ss.compute(inputs=inputs, outputs=outputs, method='local', nboot=250)
        ss.partition(2)
        inactive = ss.forward(np.random.uniform(-1, 1, 8).reshape(2, 4))[1]
        true_inactive = np.array([[1.035742, 0.046629], [0.498504, 0.371467]])
        np.testing.assert_array_almost_equal(true_inactive, inactive)

    def test_forward_03(self):
        np.random.seed(42)
        inputs = np.random.uniform(-1, 1, 60).reshape(15, 4)
        outputs = np.random.uniform(0, 5, 15)
        ss = ActiveSubspaces()
        ss.compute(inputs=inputs, outputs=outputs, method='local', nboot=250)
        ss.partition(2)
        new_inputs = np.random.uniform(-1, 1, 8).reshape(2, 4)
        active, inactive = ss.forward(new_inputs)
        reconstructed_inputs = active.dot(ss.W1.T) + inactive.dot(ss.W2.T)
        np.testing.assert_array_almost_equal(new_inputs, reconstructed_inputs)

    def test_partition_01(self):
        np.random.seed(42)
        matrix = np.random.uniform(-1, 1, 9).reshape(3, 3)
        ss = ActiveSubspaces()
        ss.evects = matrix
        ss.partition(dim=2)
        np.testing.assert_array_almost_equal(matrix[:, :2], ss.W1)

    def test_partition_02(self):
        np.random.seed(42)
        matrix = np.random.uniform(-1, 1, 9).reshape(3, 3)
        ss = ActiveSubspaces()
        ss.evects = matrix
        ss.partition(dim=2)
        np.testing.assert_array_almost_equal(matrix[:, 2:], ss.W2)

    def test_partition_03(self):
        np.random.seed(42)
        matrix = np.random.uniform(-1, 1, 9).reshape(3, 3)
        ss = ActiveSubspaces()
        ss.evects = matrix
        with self.assertRaises(TypeError):
            ss.partition(dim=2.0)

    def test_partition_04(self):
        np.random.seed(42)
        matrix = np.random.uniform(-1, 1, 9).reshape(3, 3)
        ss = ActiveSubspaces()
        ss.evects = matrix
        with self.assertRaises(ValueError):
            ss.partition(dim=0)

    def test_partition_05(self):
        np.random.seed(42)
        matrix = np.random.uniform(-1, 1, 9).reshape(3, 3)
        ss = ActiveSubspaces()
        ss.evects = matrix
        with self.assertRaises(ValueError):
            ss.partition(dim=4)

    def test_bootstrap_replicate_01(self):
        np.random.seed(42)
        matrix = np.random.uniform(-1, 1, 9).reshape(3, 3)
        weights = np.ones((3, 1)) / 3
        ss = ActiveSubspaces()
        wei = ss._bootstrap_replicate(matrix, weights)[1]
        np.testing.assert_array_almost_equal(weights, wei)

    def test_bootstrap_replicate_02(self):
        np.random.seed(42)
        matrix = np.random.uniform(-1, 1, 9).reshape(3, 3)
        weights = np.ones((3, 1)) / 3
        ss = ActiveSubspaces()
        mat = ss._bootstrap_replicate(matrix, weights)[0]
        true_matrix = np.array([[-0.88383278, 0.73235229, 0.20223002],
                                [0.19731697, -0.68796272, -0.68801096],
                                [-0.25091976, 0.90142861, 0.46398788]])
        np.testing.assert_array_almost_equal(true_matrix, mat)

    def test_compute_bootstrap_ranges_01(self):
        np.random.seed(42)
        gradients = np.random.uniform(-1, 1, 60).reshape(30, 2)
        weights = np.ones((30, 1)) / 30
        ss = ActiveSubspaces()
        ss.compute(gradients=gradients, weights=weights, nboot=100)
        true_bounds_evals = np.array([[0.3000497, 0.59008536],
                                      [0.17398718, 0.40959827]])
        np.testing.assert_array_almost_equal(true_bounds_evals, ss.evals_br)

    def test_compute_bootstrap_ranges_02(self):
        np.random.seed(42)
        gradients = np.random.uniform(-1, 1, 60).reshape(30, 2)
        weights = np.ones((30, 1)) / 30
        ss = ActiveSubspaces()
        ss.compute(gradients=gradients,
                   weights=weights,
                   method='exact',
                   nboot=100)
        true_bounds_subspace = np.array([[0.00261813, 0.58863862, 0.99998352]])
        np.testing.assert_array_almost_equal(true_bounds_subspace, ss.subs_br)

    def test_plot_eigenvalues_01(self):
        ss = ActiveSubspaces()
        with self.assertRaises(ValueError):
            ss.plot_eigenvalues(figsize=(7, 7), title='Eigenvalues')

    def test_plot_eigenvalues_02(self):
        np.random.seed(42)
        gradients = np.random.uniform(-1, 1, 200).reshape(50, 4)
        weights = np.ones((50, 1)) / 50
        ss = ActiveSubspaces()
        ss.compute(gradients=gradients, weights=weights, nboot=200)
        with assert_plot_figures_added():
            ss.plot_eigenvalues(figsize=(7, 7), title='Eigenvalues')

    def test_plot_sufficient_summary_01(self):
        ss = ActiveSubspaces()
        with self.assertRaises(ValueError):
            ss.plot_sufficient_summary(10, 10)

    def test_plot_sufficient_summary_02(self):
        np.random.seed(42)
        gradients = np.random.uniform(-1, 1, 200).reshape(50, 4)
        weights = np.ones((50, 1)) / 50
        ss = ActiveSubspaces()
        ss.compute(gradients=gradients, weights=weights, nboot=200)
        ss.partition(3)
        with self.assertRaises(ValueError):
            ss.plot_sufficient_summary(10, 10)

    def test_plot_sufficient_summary_03(self):
        np.random.seed(42)
        gradients = np.random.uniform(-1, 1, 200).reshape(50, 4)
        weights = np.ones((50, 1)) / 50
        ss = ActiveSubspaces()
        ss.compute(gradients=gradients, weights=weights, nboot=200)
        ss.partition(2)
        with assert_plot_figures_added():
            ss.plot_sufficient_summary(
                np.random.uniform(-1, 1, 100).reshape(25, 4),
                np.random.uniform(-1, 1, 25).reshape(-1, 1))
