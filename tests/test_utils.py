from unittest import TestCase
import numpy as np
from athena.utils import (Normalizer, initialize_weights, linear_program_ineq,
                          local_linear_gradients, sort_eigpairs,
                          CrossValidation, rrmse, average_rrmse)
from athena.active import ActiveSubspaces
from athena.kas import KernelActiveSubspaces
from athena.feature_map import FeatureMap


class TestUtils(TestCase):
    def test_normalizer_init_lb(self):
        normalizer = Normalizer(np.arange(5), np.arange(2, 7))
        np.testing.assert_array_equal(normalizer.lb, np.arange(5))

    def test_normalizer_init_ub(self):
        normalizer = Normalizer(np.arange(5), np.arange(2, 7))
        np.testing.assert_array_equal(normalizer.ub, np.arange(2, 7))

    def test_normalizer_fit_transform(self):
        np.random.seed(42)
        normalizer = Normalizer(-2 * np.ones(3), 4 * np.ones(3))
        inputs = np.random.uniform(-2, 4, 12).reshape(4, 3)
        ref_inputs = normalizer.fit_transform(inputs)
        true_norm = np.array([[-0.25091976, 0.90142861, 0.46398788],
                              [0.19731697, -0.68796272, -0.68801096],
                              [-0.88383278, 0.73235229, 0.20223002],
                              [0.41614516, -0.95883101, 0.9398197]])
        np.testing.assert_array_almost_equal(true_norm, ref_inputs)

    def test_normalizer_inverse_transform(self):
        np.random.seed(42)
        normalizer = Normalizer(-2 * np.ones(3), 4 * np.ones(3))
        ref_inputs = np.array([-1, 0, 1])
        inputs = normalizer.inverse_transform(ref_inputs)
        true_unnorm = np.array([-2, 1, 4])
        np.testing.assert_array_equal(true_unnorm, inputs)

    def test_initialize_weights(self):
        matrix = np.random.uniform(-1, 1, 9).reshape(3, 3)
        weights = initialize_weights(matrix)
        true_weights = np.array([0.333333, 0.333333, 0.333333]).reshape(3, 1)
        np.testing.assert_array_almost_equal(true_weights, weights)

    def test_linear_program_ineq(self):
        c = np.ones((2, 1))
        A = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        b = np.array([[0.1], [0.1], [0.1]])
        x = linear_program_ineq(c, A, b)
        true_x = np.array([0.1, 0.1]).reshape(2, 1)
        np.testing.assert_array_almost_equal(true_x, x)

    def test_local_linear_gradients_01(self):
        np.random.seed(42)
        inputs = np.random.uniform(-1.0, 1.0, size=(200, 2))
        outputs = 2 - 5 * inputs[:, 0] + 4 * inputs[:, 1]
        gradients = local_linear_gradients(inputs, outputs)[0]
        M = gradients.shape[0]
        np.testing.assert_array_almost_equal(gradients,
                                             np.tile(np.array([-5.0, 4.0]),
                                                     (M, 1)),
                                             decimal=9)

    def test_local_linear_gradients_02(self):
        np.random.seed(42)
        inputs = np.random.uniform(-1.0, 1.0, size=(200, 2))
        outputs = 2 - 5 * inputs[:, 0] + 4 * inputs[:, 1]
        gradients = local_linear_gradients(inputs, outputs, n_neighbors=8)[0]
        M = gradients.shape[0]
        np.testing.assert_array_almost_equal(gradients,
                                             np.tile(np.array([-5.0, 4.0]),
                                                     (M, 1)),
                                             decimal=9)

    def test_local_linear_gradients_03(self):
        np.random.seed(42)
        inputs = np.random.uniform(-1.0, 1.0, size=(5, 6))
        outputs = 2 - 5 * inputs[:, 0] + 4 * inputs[:, 1]
        with self.assertRaises(ValueError):
            local_linear_gradients(inputs, outputs)

    def test_local_linear_gradients_04(self):
        np.random.seed(42)
        inputs = np.random.uniform(-1.0, 1.0, size=(10, 2))
        outputs = 2 - 5 * inputs[:, 0] + 4 * inputs[:, 1]
        with self.assertRaises(TypeError):
            local_linear_gradients(inputs, outputs, n_neighbors=8.0)

    def test_local_linear_gradients_05(self):
        np.random.seed(42)
        inputs = np.random.uniform(-1.0, 1.0, size=(10, 2))
        outputs = 2 - 5 * inputs[:, 0] + 4 * inputs[:, 1]
        with self.assertRaises(ValueError):
            local_linear_gradients(inputs, outputs, n_neighbors=15)

    def test_sort_eigpairs_evals(self):
        np.random.seed(42)
        matrix = np.random.uniform(-1, 1, 9).reshape(3, 3)
        evals = sort_eigpairs(*np.linalg.eigh(matrix))[0]
        true_evals = np.array([1.466942, 1.025235, 0.294945])
        np.testing.assert_array_almost_equal(true_evals, evals)

    def test_sort_eigpairs_evects(self):
        np.random.seed(42)
        matrix = np.random.uniform(-1, 1, 9).reshape(3, 3)
        evects = sort_eigpairs(*np.linalg.eigh(matrix))[1]
        true_evects = np.array([[0.511409, 0.515297, 0.687699],
                                [-0.653819, -0.286001, 0.700517],
                                [0.557657, -0.807881, 0.190647]])
        np.testing.assert_array_almost_equal(true_evects, evects)

    def test_cross_validation_01(self):
        np.random.seed(42)
        inputs = np.random.uniform(-1, 1, 15).reshape(5, -1)
        outputs = np.random.uniform(0, 5, 10).reshape(5, -1)
        gradients = np.random.uniform(-1, 1, 30).reshape(5, 2, 3)
        ss = ActiveSubspaces(dim=1)
        cv = CrossValidation(inputs, outputs, gradients, ss)
        np.testing.assert_array_almost_equal(cv.inputs, inputs)

    def test_cross_validation_02(self):
        np.random.seed(42)
        inputs = np.random.uniform(-1, 1, 15).reshape(5, -1)
        outputs = np.random.uniform(0, 5, 10).reshape(5, -1)
        gradients = np.random.uniform(-1, 1, 30).reshape(5, 2, 3)
        ss = ActiveSubspaces(dim=1, method='exact')
        cv = CrossValidation(inputs, outputs, gradients, ss)
        np.testing.assert_array_almost_equal(cv.outputs, outputs)

    def test_cross_validation_03(self):
        np.random.seed(42)
        inputs = np.random.uniform(-1, 1, 15).reshape(5, -1)
        outputs = np.random.uniform(0, 5, 10).reshape(5, -1)
        gradients = np.random.uniform(-1, 1, 30).reshape(5, 2, 3)
        ss = ActiveSubspaces(dim=1)
        cv = CrossValidation(inputs, outputs, gradients, ss)
        np.testing.assert_array_almost_equal(cv.gradients, gradients)

    def test_cross_validation_04(self):
        np.random.seed(42)
        inputs = np.random.uniform(-1, 1, 15).reshape(5, -1)
        outputs = np.random.uniform(0, 5, 10).reshape(5, -1)
        gradients = np.random.uniform(-1, 1, 30).reshape(5, 2, 3)
        ss = ActiveSubspaces(dim=1, method='exact')
        cv = CrossValidation(inputs, outputs, gradients, ss)
        self.assertEqual(cv.ss, ss)

    def test_cross_validation_05(self):
        np.random.seed(42)
        inputs = np.random.uniform(-1, 1, 15).reshape(5, -1)
        outputs = np.random.uniform(0, 5, 10).reshape(5, -1)
        gradients = np.random.uniform(-1, 1, 30).reshape(5, 2, 3)
        ss = ActiveSubspaces(dim=1)
        cv = CrossValidation(inputs, outputs, gradients, ss)
        self.assertEqual(cv.folds, 5)

    def test_cross_validation_06(self):
        np.random.seed(42)
        inputs = np.random.uniform(-1, 1, 15).reshape(5, -1)
        outputs = np.random.uniform(0, 5, 10).reshape(5, -1)
        gradients = np.random.uniform(-1, 1, 30).reshape(5, 2, 3)
        ss = ActiveSubspaces(dim=1)
        cv = CrossValidation(inputs, outputs, gradients, ss)
        self.assertIsNone(cv.gp)

    def test_cross_validation_07(self):
        np.random.seed(42)
        inputs = np.random.uniform(-1, 1, 15).reshape(5, -1)
        outputs = np.random.uniform(0, 5, 10).reshape(5, -1)
        gradients = np.random.uniform(-1, 1, 30).reshape(5, 2, 3)
        with self.assertRaises(ValueError):
            CrossValidation(inputs, outputs, None, None)

    def test_cross_validation_fit_01(self):
        np.random.seed(42)
        inputs = np.random.uniform(-1, 1, 15).reshape(5, -1)
        outputs = np.random.uniform(0, 5, 10).reshape(5, -1)
        gradients = np.random.uniform(-1, 1, 30).reshape(5, 2, 3)
        ss = ActiveSubspaces(dim=2, method='exact')
        csv = CrossValidation(inputs=inputs,
                              outputs=outputs,
                              gradients=gradients,
                              folds=3,
                              subspace=ss)
        csv.fit(inputs, gradients, outputs)
        self.assertEqual(csv.gp.X_train_.shape[1], 2)

    def test_cross_validation_fit_02(self):
        np.random.seed(42)
        inputs = np.random.uniform(-1, 1, 15).reshape(5, -1)
        outputs = np.random.uniform(0, 5, 10).reshape(5, -1)
        gradients = np.random.uniform(-1, 1, 30).reshape(5, 2, 3)
        ss = ActiveSubspaces(dim=1)
        csv = CrossValidation(inputs=inputs,
                              outputs=outputs,
                              gradients=gradients,
                              folds=3,
                              subspace=ss)
        csv.fit(inputs, gradients, outputs)
        self.assertIsNotNone(csv.gp)

    def test_cross_validation_run_01(self):
        np.random.seed(42)
        inputs = np.random.uniform(-1, 1, 15).reshape(5, -1)
        outputs = np.random.uniform(0, 5, 10).reshape(5, -1)
        gradients = np.random.uniform(-1, 1, 30).reshape(5, 2, 3)
        ss = ActiveSubspaces(dim=1, method='exact')
        csv = CrossValidation(inputs=inputs,
                              outputs=outputs,
                              gradients=gradients,
                              folds=2,
                              subspace=ss)
        true_value = (9.696572, 6.45413)
        np.testing.assert_array_almost_equal(csv.run(), true_value)

    def test_cross_validation_run_02(self):
        np.random.seed(42)
        inputs = np.random.uniform(-1, 1, 10).reshape(5, 2)
        outputs = np.random.uniform(0, 5, 10).reshape(5, 2)
        gradients = np.random.uniform(-1, 1, 20).reshape(5, 2, 2)
        fm = FeatureMap(distr='laplace',
                        bias=np.random.uniform(-1, 1, 3),
                        input_dim=2,
                        n_features=3,
                        params=np.zeros(1),
                        sigma_f=outputs.var())
        ss = KernelActiveSubspaces(dim=1, feature_map=fm)
        csv = CrossValidation(inputs=inputs,
                              outputs=outputs,
                              gradients=gradients,
                              folds=2,
                              subspace=ss)
        true_value = (2.362181, 0.41159)
        np.testing.assert_array_almost_equal(csv.run(), true_value)

    def test_rrmse_01(self):
        np.random.seed(42)
        predictions = np.random.uniform(-1, 1, 5).reshape(5)
        targets = np.random.uniform(-1, 1, 5).reshape(5)
        true = 1.5298111757191089
        np.testing.assert_array_equal(rrmse(predictions, targets), true)

    def test_rrmse_02(self):
        np.random.seed(42)
        predictions = np.random.uniform(-1, 1, 10).reshape(5, 2)
        targets = np.random.uniform(-1, 1, 10).reshape(5, 2)
        true = 0.9089760363050161
        np.testing.assert_array_equal(rrmse(predictions, targets), true)

    def test_rrmse_03(self):
        np.random.seed(42)
        predictions = np.random.uniform(-1, 1, 10).reshape(5, 2)
        targets = np.random.uniform(-1, 1, 10).reshape(2, 5)
        with self.assertRaises(ValueError):
            rrmse(predictions, targets)

    def test_average_rrmse_01(self):
        np.random.seed(42)
        inputs = np.random.uniform(-1, 1, 10).reshape(5, 2)
        outputs = np.random.uniform(0, 5, 10).reshape(5, 2)
        gradients = np.random.uniform(-1, 1, 20).reshape(5, 2, 2)
        fm = FeatureMap(distr='laplace',
                        bias=np.random.uniform(-1, 1, 3),
                        input_dim=2,
                        n_features=3,
                        params=np.zeros(1),
                        sigma_f=outputs.var())
        ss = KernelActiveSubspaces(dim=1, feature_map=fm)
        csv = CrossValidation(inputs=inputs,
                              outputs=outputs,
                              gradients=gradients,
                              folds=2,
                              subspace=ss)
        best = [0.1, np.zeros((3, 2))]
        hyperparams = np.array([-1.])
        score = average_rrmse(hyperparams, best, csv, verbose=False, resample=1)
        np.testing.assert_equal(best[0], 0.1)

    def test_average_rrmse_02(self):
        np.random.seed(42)
        inputs = np.random.uniform(-1, 1, 10).reshape(5, 2)
        outputs = np.random.uniform(0, 5, 10).reshape(5, 2)
        gradients = np.random.uniform(-1, 1, 20).reshape(5, 2, 2)
        fm = FeatureMap(distr='laplace',
                        bias=np.random.uniform(-1, 1, 3),
                        input_dim=2,
                        n_features=3,
                        params=np.zeros(1),
                        sigma_f=outputs.var())
        ss = KernelActiveSubspaces(dim=1, feature_map=fm, method='exact')
        csv = CrossValidation(inputs=inputs,
                              outputs=outputs,
                              gradients=gradients,
                              folds=2,
                              subspace=ss)
        best = [0.1, np.zeros((3, 2))]
        hyperparams = np.array([-1.])
        score = average_rrmse(hyperparams, best, csv, verbose=True, resample=1)
        true = 7.409494
        np.testing.assert_array_almost_equal(score, true)
