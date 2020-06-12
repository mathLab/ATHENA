from unittest import TestCase
from athena import ProjectionFactory
import numpy as np

class TestProjectionFactory(TestCase):
    def test_beta(self):
        np.random.seed(42)
        projection = ProjectionFactory('beta')
        value = projection(input_dim=2, n_features=3, params=[2.1, 1.8])
        true_value = np.array([[0.6580682, 0.5509215],
                               [0.6484241, 0.3526883],
                               [0.9312698, 0.2805237]])
        np.testing.assert_array_almost_equal(true_value, value)

    def test_cauchy(self):
        np.random.seed(42)
        projection = ProjectionFactory('cauchy')
        value = projection(n_features=3, params=3.4)
        true_value = np.array([2.984093, 9.80677 , 4.292814])
        np.testing.assert_array_almost_equal(true_value, value)

    def test_dirichlet(self):
        np.random.seed(42)
        projection = ProjectionFactory('dirichlet')
        value = projection(input_dim=2, n_features=3, params=[2.7, 2.0])
        true_value = np.array([[0.598184, 0.401816],
                               [0.499997, 0.500003],
                               [0.603821, 0.396179]])
        np.testing.assert_array_almost_equal(true_value, value)

    def test_laplace(self):
        np.random.seed(42)
        projection = ProjectionFactory('laplace')
        value = projection(input_dim=2, n_features=3, params=[4.1, 0.3])
        true_value = np.array([[4.013327, 4.795092],
                               [4.28708 , 4.165939],
                               [3.75061 , 3.750564]])
        np.testing.assert_array_almost_equal(true_value, value)

    def test_multivariate_normal(self):
        np.random.seed(42)
        projection = ProjectionFactory('multivariate_normal')
        value = projection(input_dim=2, n_features=3, params=[2.5, 0.8])
        true_value = np.array([[ 0.785374, -0.123667],
                               [ 1.024085,  1.362239],
                               [-0.370229, -0.209418]])
        np.testing.assert_array_almost_equal(true_value, value)

    def test_normal(self):
        np.random.seed(42)
        projection = ProjectionFactory('normal')
        value = projection(input_dim=2, n_features=3, params=[0.6, 1.0])
        true_value = np.array([[ 0.298028, -0.082959],
                               [ 0.388613,  0.913818],
                               [-0.140492, -0.140482]])
        np.testing.assert_array_almost_equal(true_value, value)

    def test_uniform(self):
        np.random.seed(42)
        projection = ProjectionFactory('uniform')
        value = projection(input_dim=2, n_features=3, params=[1.1, 1.2])
        true_value = np.array([[1.137454, 1.195071],
                               [1.173199, 1.159866],
                               [1.115602, 1.115599]])
        np.testing.assert_array_almost_equal(true_value, value)