
from unittest import TestCase
from athena.utils import initialize_weights, local_linear_gradients, sort_eigpairs
import numpy as np

class TestUtils(TestCase):

    def test_initialize_weights(self):
        matrix = np.random.uniform(-1, 1, 9).reshape(3, 3)
        weights = initialize_weights(matrix)
        true_weights = np.array([0.333333, 0.333333, 0.333333]).reshape(3, 1)
        np.testing.assert_array_almost_equal(true_weights, weights)

    def test_sort_eigpairs_evals(self):
        np.random.seed(42)
        matrix = np.random.uniform(-1, 1, 9).reshape(3, 3)
        evals = sort_eigpairs(matrix)[0]
        true_evals = np.array([[1.466942], [1.025235], [0.294945]])
        np.testing.assert_array_almost_equal(true_evals, evals)

    def test_sort_eigpairs_evects(self):
        np.random.seed(42)
        matrix = np.random.uniform(-1, 1, 9).reshape(3, 3)
        evects = sort_eigpairs(matrix)[1]
        true_evects = np.array([[ 0.511409,  0.515297,  0.687699],
                                [-0.653819, -0.286001,  0.700517],
                                [ 0.557657, -0.807881,  0.190647]])
        np.testing.assert_array_almost_equal(true_evects, evects)

    def test_local_linear_gradients_01(self):
        np.random.seed(42)
        inputs = np.random.uniform(-1.0, 1.0, size=(200, 2))
        outputs = 2 - 5*inputs[:, 0] + 4*inputs[:, 1]
        gradients = local_linear_gradients(inputs, outputs)
        M = gradients.shape[0]
        np.testing.assert_array_almost_equal(gradients, np.tile(np.array([-5.0, 4.0]), (M, 1)), decimal=9)

    def test_local_linear_gradients_02(self):
        np.random.seed(42)
        inputs = np.random.uniform(-1.0, 1.0, size=(200, 2))
        outputs = 2 - 5*inputs[:, 0] + 4*inputs[:, 1]
        gradients = local_linear_gradients(inputs, outputs, n_neighbors=8)
        M = gradients.shape[0]
        np.testing.assert_array_almost_equal(gradients, np.tile(np.array([-5.0, 4.0]), (M, 1)), decimal=9)
