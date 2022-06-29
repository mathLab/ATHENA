from unittest import TestCase
import numpy as np
from athena.subspaces import Subspaces


class TestUtils(TestCase):
    def test_init_W1(self):
        ss = Subspaces(dim=1)
        self.assertIsNone(ss.W1)

    def test_init_W2(self):
        ss = Subspaces(dim=1)
        self.assertIsNone(ss.W2)

    def test_init_evals(self):
        ss = Subspaces(dim=1)
        self.assertIsNone(ss.evals)

    def test_init_evects(self):
        ss = Subspaces(dim=1)
        self.assertIsNone(ss.evects)

    def test_init_evals_br(self):
        ss = Subspaces(dim=1)
        self.assertIsNone(ss.evals_br)

    def test_init_subs_br(self):
        ss = Subspaces(dim=1)
        self.assertIsNone(ss.subs_br)

    def test_init_dim(self):
        ss = Subspaces(dim=1)
        self.assertEqual(ss.dim, 1)

    def test_fit(self):
        ss = Subspaces(dim=1)
        with self.assertRaises(NotImplementedError):
            ss.fit()

    def test_transform(self):
        ss = Subspaces(dim=1)
        with self.assertRaises(NotImplementedError):
            ss.transform(42)

    def test_inverse_transform(self):
        ss = Subspaces(dim=1)
        with self.assertRaises(NotImplementedError):
            ss.inverse_transform(10, 10)

    def test_partition_01(self):
        np.random.seed(42)
        matrix = np.random.uniform(-1, 1, 9).reshape(3, 3)
        ss = Subspaces(dim=2)
        ss.evects = matrix
        ss._partition()
        np.testing.assert_array_almost_equal(matrix[:, :2], ss.W1)

    def test_partition_02(self):
        np.random.seed(42)
        matrix = np.random.uniform(-1, 1, 9).reshape(3, 3)
        ss = Subspaces(dim=2)
        ss.evects = matrix
        ss._partition()
        np.testing.assert_array_almost_equal(matrix[:, 2:], ss.W2)

    def test_partition_03(self):
        np.random.seed(42)
        matrix = np.random.uniform(-1, 1, 9).reshape(3, 3)
        ss = Subspaces(dim=2.0)
        ss.evects = matrix
        with self.assertRaises(ValueError):
            ss._partition()

    def test_partition_04(self):
        np.random.seed(42)
        matrix = np.random.uniform(-1, 1, 9).reshape(3, 3)
        ss = Subspaces(dim=0)
        ss.evects = matrix
        with self.assertRaises(ValueError):
            ss._partition()

    def test_partition_05(self):
        np.random.seed(42)
        matrix = np.random.uniform(-1, 1, 9).reshape(3, 3)
        ss = Subspaces(dim=4)
        ss.evects = matrix
        with self.assertRaises(ValueError):
            ss._partition()

    def test_bootstrap_replicate_01(self):
        np.random.seed(42)
        matrix = np.random.uniform(-1, 1, 9).reshape(3, 3)
        weights = np.ones((3, 1)) / 3
        ss = Subspaces(dim=1)
        wei = ss._bootstrap_replicate(matrix, weights)[1]
        np.testing.assert_array_almost_equal(weights, wei)

    def test_bootstrap_replicate_02(self):
        np.random.seed(42)
        matrix = np.random.uniform(-1, 1, 9).reshape(3, 3)
        weights = np.ones((3, 1)) / 3
        ss = Subspaces(dim=1)
        mat = ss._bootstrap_replicate(matrix, weights)[0]
        true_matrix = np.array([[-0.88383278, 0.73235229, 0.20223002],
                                [0.19731697, -0.68796272, -0.68801096],
                                [-0.25091976, 0.90142861, 0.46398788]])
        np.testing.assert_array_almost_equal(true_matrix, mat)

    def test_plot_eigenvalues(self):
        ss = Subspaces(dim=1)
        with self.assertRaises(TypeError):
            ss.plot_eigenvalues(figsize=(7, 7), title='Eigenvalues')

    def test_plot_eigenvectors(self):
        ss = Subspaces(dim=1)
        with self.assertRaises(TypeError):
            ss.plot_eigenvectors(n_evects=2, title='Eigenvectors')

    def test_plot_sufficient_summary(self):
        ss = Subspaces(dim=1)
        inputs = np.diag(np.ones(3))
        outputs = np.ones(3).reshape(3, 1)
        with self.assertRaises(TypeError):
            ss.plot_sufficient_summary(inputs,
                                       outputs,
                                       figsize=(7, 7),
                                       title='Sufficient_summary_plots')

    def test_partition_spectral_gap(self):
        np.random.seed(42)
        matrix = np.array([[1, 1, 1], [2, -4.5, 2], [1, 1.1, 1]])
        weights = np.ones((3, 1))
        ss = Subspaces(dim=0)
        ss.evals, ss.evects = ss._build_decompose_cov_matrix(matrix, weights)
        dim = ss._set_dim_spectral_gap()
        self.assertEqual(dim, 1)

    def test_partition_residual_energy(self):
        np.random.seed(42)
        matrix = np.array([[1, 1, 1], [2, -4.5, 2], [1, 1.1, 1]])
        weights = np.ones((3, 1))
        ss = Subspaces(dim=1)
        ss.evals, ss.evects = ss._build_decompose_cov_matrix(matrix, weights)
        dim = ss._set_dim_residual_energy()
        self.assertEqual(dim, 1)
