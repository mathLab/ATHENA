from unittest import TestCase
import numpy as np
from athena.active import ActiveSubspaces
from contextlib import contextmanager
import matplotlib.pyplot as plt


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
        ss = ActiveSubspaces(dim=1)
        self.assertIsNone(ss.W1)

    def test_init_W2(self):
        ss = ActiveSubspaces(dim=1)
        self.assertIsNone(ss.W2)

    def test_init_evals(self):
        ss = ActiveSubspaces(dim=1)
        self.assertIsNone(ss.evals)

    def test_init_evects(self):
        ss = ActiveSubspaces(dim=1)
        self.assertIsNone(ss.evects)

    def test_init_evals_br(self):
        ss = ActiveSubspaces(dim=1)
        self.assertIsNone(ss.evals_br)

    def test_init_subs_br(self):
        ss = ActiveSubspaces(dim=1)
        self.assertIsNone(ss.subs_br)

    def test_init_dim(self):
        ss = ActiveSubspaces(dim=1)
        self.assertEqual(ss.dim, 1)

    def test_fit_01(self):
        ss = ActiveSubspaces(dim=1)
        with self.assertRaises(TypeError):
            ss.fit()

    def test_fit_02(self):
        np.random.seed(42)
        gradients = np.random.uniform(-1, 1, 60).reshape(15, 4)
        weights = np.ones((15, 1)) / 15
        ss = ActiveSubspaces(dim=1, method='exact', n_boot=150)
        ss.fit(gradients=gradients, weights=weights)
        true_evals = np.array([0.571596, 0.465819, 0.272198, 0.175012])
        np.testing.assert_array_almost_equal(true_evals, ss.evals)

    def test_fit_03(self):
        np.random.seed(42)
        gradients = np.random.uniform(-1, 1, 60).reshape(15, 4)
        weights = np.ones((15, 1)) / 15
        ss = ActiveSubspaces(dim=1, n_boot=200)
        ss.fit(gradients=gradients, weights=weights)
        true_evects = np.array([[0.019091, -0.408566, -0.861223, -0.301669],
                                [0.767799, 0.199069, -0.268823, 0.546434],
                                [0.463451, -0.758442, 0.427696, -0.164486],
                                [0.441965, 0.467131, 0.055723, -0.763774]])
        np.testing.assert_array_almost_equal(true_evects, ss.evects)

    def test_fit_04(self):
        np.random.seed(42)
        inputs = np.random.uniform(-1, 1, 60).reshape(15, 4)
        outputs = np.random.uniform(0, 5, 15)
        ss = ActiveSubspaces(dim=1, method='local', n_boot=150)
        ss.fit(inputs=inputs, outputs=outputs)
        true_evals = np.array([13.794711, 11.102377, 3.467318, 1.116324])
        np.testing.assert_array_almost_equal(true_evals, ss.evals)

    def test_fit_05(self):
        np.random.seed(42)
        inputs = np.random.uniform(-1, 1, 60).reshape(15, 4)
        outputs = np.random.uniform(0, 5, 15)
        ss = ActiveSubspaces(dim=1, method='local', n_boot=200)
        ss.fit(inputs=inputs, outputs=outputs)
        true_evects = np.array([[0.164383, 0.717021, 0.237246, -0.634486],
                                [0.885808, 0.177628, -0.004112, 0.428691],
                                [0.255722, -0.558199, 0.734083, -0.290071],
                                [0.350612, -0.377813, -0.636254, -0.574029]])
        np.testing.assert_array_almost_equal(true_evects, ss.evects)

    def test_fit_06(self):
        np.random.seed(42)
        gradients = np.random.uniform(-1, 1, 180).reshape(15, 3, 4)
        weights = np.ones((15, 1)) / 15
        ss = ActiveSubspaces(dim=1, n_boot=150)
        ss.fit(gradients=gradients, weights=weights)
        true_evals = np.array([1.32606312, 1.20519582, 0.94811868, 0.68505712])
        np.testing.assert_array_almost_equal(true_evals, ss.evals)

    def test_fit_07(self):
        np.random.seed(42)
        gradients = np.random.uniform(-1, 1, 180).reshape(15, 3, 4)
        weights = np.ones((15, 1)) / 15
        ss = ActiveSubspaces(dim=1, method='exact', n_boot=150)
        ss.fit(gradients=gradients, weights=weights)
        true_evects = np.array(
            [[0.67237041, 0.49917148, 0.50889687, 0.1994238],
             [0.20398894, -0.66183856, 0.09970486, 0.71443486],
             [-0.52895262, -0.11348076, 0.83802337, -0.07104981],
             [0.47593663, -0.54764923, 0.16970489, -0.66690696]])
        np.testing.assert_array_almost_equal(true_evects, ss.evects)

    def test_fit_08(self):
        np.random.seed(42)
        gradients = np.random.uniform(-1, 1, 180).reshape(15, 3, 4)
        weights = np.ones((15, 1)) / 15
        metric = np.diag(2 * np.ones(3))
        ss = ActiveSubspaces(dim=1, method='exact', n_boot=150)
        ss.fit(gradients=gradients, weights=weights, metric=metric)
        true_evects = np.array(
            [[0.67237041, 0.49917148, 0.50889687, 0.1994238],
             [0.20398894, -0.66183856, 0.09970486, 0.71443486],
             [-0.52895262, -0.11348076, 0.83802337, -0.07104981],
             [0.47593663, -0.54764923, 0.16970489, -0.66690696]])
        np.testing.assert_array_almost_equal(true_evects, ss.evects)

    def test_fit_09(self):
        np.random.seed(42)
        inputs = np.random.uniform(-1, 1, 60).reshape(15, 4)
        outputs = np.random.uniform(0, 5, 45).reshape(15, 3)
        weights = np.ones((15, 1)) / 15
        ss = ActiveSubspaces(dim=1, method='local', n_boot=150)
        ss.fit(inputs=inputs, outputs=outputs, weights=weights)
        true_evals = np.array(
            [84.08055975, 25.87980349, 9.61982202, 8.29248646])
        np.testing.assert_array_almost_equal(true_evals, ss.evals)

    def test_fit_10(self):
        np.random.seed(42)
        inputs = np.random.uniform(-1, 1, 60).reshape(15, 4)
        outputs = np.random.uniform(0, 5, 45).reshape(15, 3)
        weights = np.ones((15, 1)) / 15
        metric = np.diag(2 * np.ones(3))
        ss = ActiveSubspaces(dim=1, method='local', n_boot=150)
        ss.fit(inputs=inputs, outputs=outputs, weights=weights, metric=metric)
        true_evects = np.array(
            [[0.75159386, 0.34814972, 0.5093598, 0.23334745],
             [-0.44047075, 0.89306499, -0.00249322, 0.09172904],
             [0.35007644, 0.2587911, -0.30548878, -0.84684725],
             [0.34429445, 0.11938954, -0.8045017, 0.46902504]])
        np.testing.assert_array_almost_equal(true_evects, ss.evects)

    def test_fit_11(self):
        np.random.seed(42)
        grad = np.array(
            [[-0.50183952, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [-1.26638196, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0.43017941, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0.65008914, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        gradients = (grad[i, :] for i in range(4))
        ss = ActiveSubspaces(dim=2, method='exact', n_boot=150)
        ss.fit(gradients=gradients)
        true_evals = [2.040621, 0.]
        np.testing.assert_array_almost_equal(true_evals, ss.evals)

    def test_fit_12(self):
        np.random.seed(42)
        grad = np.array(
            [[-0.50183952, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [-1.26638196, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0.43017941, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0.65008914, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        gradients = (grad[i, :] for i in range(4))
        ss = ActiveSubspaces(dim=2, method='exact', n_boot=150)
        ss.fit(gradients=gradients)
        true_evects = np.array([[1., 0.], [0., 1.], [0., 0.], [0.,
                                                               0.], [0., 0.],
                                [0., 0.], [0., 0.], [0., 0.], [0.,
                                                               0.], [0., 0.],
                                [0., 0.], [0., 0.], [0., 0.], [0., 0.],
                                [0., 0.]])
        np.testing.assert_array_almost_equal(true_evects, ss.evects)

    def test_activity_scores_01(self):
        np.random.seed(42)
        gradients = np.random.uniform(-1, 1, 180).reshape(15, 3, 4)
        ss = ActiveSubspaces(dim=1, method='exact', n_boot=150)
        ss.fit(gradients=gradients)
        true_scores = np.array([0.599489, 0.055179, 0.37102, 0.300374])
        np.testing.assert_array_almost_equal(true_scores, ss.activity_scores)

    def test_activity_scores_02(self):
        np.random.seed(42)
        gradients = np.random.uniform(-1, 1, 180).reshape(15, 3, 4)
        ss = ActiveSubspaces(dim=2, method='exact', n_boot=150)
        ss.fit(gradients=gradients)
        true_scores = np.array([0.89979047, 0.58309172, 0.38654072, 0.6618360])
        np.testing.assert_array_almost_equal(true_scores, ss.activity_scores)

    def test_activity_scores_03(self):
        np.random.seed(42)
        gradients = np.random.uniform(-1, 1, 180).reshape(15, 3, 4)
        ss = ActiveSubspaces(dim=2, method='exact', n_boot=150)
        with self.assertRaises(TypeError):
            ss.activity_scores

    def test_activity_scores_04(self):
        np.random.seed(42)
        inputs = np.random.uniform(-1, 1, 60).reshape(4, 15)
        gradients = (inputs[i, :] for i in range(4))
        ss = ActiveSubspaces(dim=2, method='exact', n_boot=150)
        ss.fit(gradients=gradients)
        true_scores = np.array([
            0.061844, 0.914259, 0.868018, 0.830417, 1.081719, 0.652845,
            0.486826, 0.974479, 0.165313, 0.111259, 1.135542, 0.547697,
            1.099458, 0.860475, 0.377612
        ])
        np.testing.assert_array_almost_equal(true_scores, ss.activity_scores)

    def test_transform_01(self):
        np.random.seed(42)
        inputs = np.random.uniform(-1, 1, 60).reshape(15, 4)
        outputs = np.random.uniform(0, 5, 15)
        ss = ActiveSubspaces(dim=2, method='local', n_boot=250)
        ss.fit(inputs=inputs, outputs=outputs)
        active = ss.transform(np.random.uniform(-1, 1, 8).reshape(2, 4))[0]
        true_active = np.array([[0.232762, 0.419052], [0.613532, 1.004439]])
        np.testing.assert_array_almost_equal(true_active, active)

    def test_transform_02(self):
        np.random.seed(42)
        inputs = np.random.uniform(-1, 1, 60).reshape(15, 4)
        outputs = np.random.uniform(0, 5, 15)
        ss = ActiveSubspaces(dim=2, method='local', n_boot=250)
        ss.fit(inputs=inputs, outputs=outputs)
        inactive = ss.transform(np.random.uniform(-1, 1, 8).reshape(2, 4))[1]
        true_inactive = np.array([[-0.792408, -0.57175],
                                  [-0.144381,  0.043564]])
        np.testing.assert_array_almost_equal(true_inactive, inactive)

    def test_transform_03(self):
        np.random.seed(42)
        inputs = np.random.uniform(-1, 1, 60).reshape(15, 4)
        outputs = np.random.uniform(0, 5, 45).reshape(15, 3)
        ss = ActiveSubspaces(dim=2, method='local', n_boot=250)
        ss.fit(inputs=inputs, outputs=outputs, metric=np.diag(np.ones(3)))
        new_inputs = np.random.uniform(-1, 1, 8).reshape(2, 4)
        active, inactive = ss.transform(new_inputs)
        reconstructed_inputs = active.dot(ss.W1.T) + inactive.dot(ss.W2.T)
        np.testing.assert_array_almost_equal(new_inputs, reconstructed_inputs)

    def test_transform_04(self):
        np.random.seed(42)
        inputs = np.random.uniform(-1, 1, 60).reshape(15, 4)
        outputs = np.random.uniform(0, 5, 45).reshape(15, 3)
        ss = ActiveSubspaces(dim=2, method='local', n_boot=250)
        ss.fit(inputs=inputs, outputs=outputs)
        active = ss.transform(np.random.uniform(-1, 1, 8).reshape(2, 4))[0]
        true_active = np.array([[0.15284753, 0.67109407],
                                [0.69006622, -0.4165206]])
        np.testing.assert_array_almost_equal(true_active, active)

    def test_transform_05(self):
        np.random.seed(42)
        inputs = np.random.uniform(-1, 1, 60).reshape(15, 4)
        outputs = np.random.uniform(0, 5, 15)
        ss = ActiveSubspaces(dim=2, method='local', n_boot=250)
        ss.fit(inputs=inputs, outputs=outputs)
        inactive = ss.transform(np.random.uniform(-1, 1, 8).reshape(2, 4))[1]
        true_inactive = np.array([[-0.792408, -0.57175 ],
                                  [-0.144381,  0.043564]])
        np.testing.assert_array_almost_equal(true_inactive, inactive)

    def test_transform_06(self):
        np.random.seed(42)
        inputs = np.random.uniform(-1, 1, 60).reshape(15, 4)
        outputs = np.random.uniform(0, 5, 15)
        ss = ActiveSubspaces(dim=2, method='local', n_boot=250)
        ss.fit(inputs=inputs, outputs=outputs)
        ss.W2 = None
        inactive = ss.transform(np.random.uniform(-1, 1, 8).reshape(2, 4))[1]
        self.assertIsNone(inactive)

    def test_inverse_transform_01(self):
        np.random.seed(42)
        inputs = np.random.uniform(-1, 1, 60).reshape(15, 4)
        outputs = np.random.uniform(0, 5, 15)
        ss = ActiveSubspaces(dim=1, method='local', n_boot=250)
        ss.fit(inputs=inputs, outputs=outputs)
        new_inputs = np.random.uniform(-1, 1, 8).reshape(2, 4)
        active = ss.transform(new_inputs)[0]
        new_inputs = ss.inverse_transform(reduced_inputs=active, n_points=5)[0]
        np.testing.assert_array_almost_equal(np.kron(active, np.ones((5, 1))),
                                             new_inputs.dot(ss.W1))

    def test_inverse_transform_02(self):
        np.random.seed(42)
        inputs = np.random.uniform(-1, 1, 80).reshape(16, 5)
        outputs = np.random.uniform(-1, 3, 16)
        ss = ActiveSubspaces(dim=2, method='local', n_boot=250)
        ss.fit(inputs=inputs, outputs=outputs)
        new_inputs = np.random.uniform(-1, 1, 15).reshape(3, 5)
        active = ss.transform(new_inputs)[0]
        new_inputs = ss.inverse_transform(reduced_inputs=active,
                                          n_points=500)[0]
        np.testing.assert_array_almost_equal(np.kron(active, np.ones((500, 1))),
                                             new_inputs.dot(ss.W1))

    def test_rejection_sampling_inactive_01(self):
        np.random.seed(42)
        inputs = np.random.uniform(-1, 1, 60).reshape(15, 4)
        outputs = np.random.uniform(0, 5, 15)
        ss = ActiveSubspaces(dim=1, method='local', n_boot=250)
        ss.fit(inputs=inputs, outputs=outputs)
        new_inputs = np.random.uniform(-1, 1, 8).reshape(2, 4)
        active = ss.transform(new_inputs)[0]
        inactive_swap = np.array([
            ss._rejection_sampling_inactive(reduced_input=red_inp, n_points=1)
            for red_inp in active
        ])
        inactive_inputs = np.swapaxes(inactive_swap, 1, 2)
        new_inputs = ss._rotate_x(reduced_inputs=active,
                                  inactive_inputs=inactive_inputs)[0]
        np.testing.assert_array_almost_equal(active, new_inputs.dot(ss.W1))

    def test_rejection_sampling_inactive_02(self):
        np.random.seed(42)
        inputs = np.random.uniform(-1, 1, 60).reshape(15, 4)
        outputs = np.random.uniform(0, 5, 15)
        ss = ActiveSubspaces(dim=1, method='local', n_boot=250)
        ss.fit(inputs=inputs, outputs=outputs)
        new_inputs = np.random.uniform(-1, 1, 8).reshape(2, 4)
        active = ss.transform(new_inputs)[0]
        inactive_swap = np.array([
            ss._rejection_sampling_inactive(reduced_input=red_inp, n_points=10)
            for red_inp in active
        ])
        inactive_inputs = np.swapaxes(inactive_swap, 1, 2)
        new_inputs = ss._rotate_x(reduced_inputs=active,
                                  inactive_inputs=inactive_inputs)[0]
        np.testing.assert_array_almost_equal(np.kron(active, np.ones((10, 1))),
                                             new_inputs.dot(ss.W1))

    def test_hit_and_run_inactive_01(self):
        np.random.seed(42)
        inputs = np.random.uniform(-1, 1, 60).reshape(15, 4)
        outputs = np.random.uniform(0, 5, 15)
        ss = ActiveSubspaces(dim=1, method='local', n_boot=250)
        ss.fit(inputs=inputs, outputs=outputs)
        new_inputs = np.random.uniform(-1, 1, 8).reshape(2, 4)
        active = ss.transform(new_inputs)[0]
        inactive_swap = np.array([
            ss._hit_and_run_inactive(reduced_input=red_inp, n_points=1)
            for red_inp in active
        ])
        inactive_inputs = np.swapaxes(inactive_swap, 1, 2)
        new_inputs = ss._rotate_x(reduced_inputs=active,
                                  inactive_inputs=inactive_inputs)[0]
        np.testing.assert_array_almost_equal(active, new_inputs.dot(ss.W1))

    def test_hit_and_run_inactive_02(self):
        np.random.seed(42)
        inputs = np.random.uniform(-1, 1, 60).reshape(15, 4)
        outputs = np.random.uniform(0, 5, 15)
        ss = ActiveSubspaces(dim=1, method='local', n_boot=250)
        ss.fit(inputs=inputs, outputs=outputs)
        new_inputs = np.random.uniform(-1, 1, 8).reshape(2, 4)
        active = ss.transform(new_inputs)[0]
        inactive_swap = np.array([
            ss._hit_and_run_inactive(reduced_input=red_inp, n_points=10)
            for red_inp in active
        ])
        inactive_inputs = np.swapaxes(inactive_swap, 1, 2)
        new_inputs = ss._rotate_x(reduced_inputs=active,
                                  inactive_inputs=inactive_inputs)[0]
        np.testing.assert_array_almost_equal(np.kron(active, np.ones((10, 1))),
                                             new_inputs.dot(ss.W1))

    def test_partition_01(self):
        np.random.seed(42)
        matrix = np.random.uniform(-1, 1, 9).reshape(3, 3)
        ss = ActiveSubspaces(dim=2)
        ss.evects = matrix
        ss._partition()
        np.testing.assert_array_almost_equal(matrix[:, :2], ss.W1)

    def test_partition_02(self):
        np.random.seed(42)
        matrix = np.random.uniform(-1, 1, 9).reshape(3, 3)
        ss = ActiveSubspaces(dim=2)
        ss.evects = matrix
        ss._partition()
        np.testing.assert_array_almost_equal(matrix[:, 2:], ss.W2)

    def test_partition_03(self):
        np.random.seed(42)
        matrix = np.random.uniform(-1, 1, 9).reshape(3, 3)
        ss = ActiveSubspaces(dim=2.0)
        ss.evects = matrix
        with self.assertRaises(ValueError):
            ss._partition()

    def test_partition_04(self):
        np.random.seed(42)
        matrix = np.random.uniform(-1, 1, 9).reshape(3, 3)
        ss = ActiveSubspaces(dim=0.)
        ss.evects = matrix
        with self.assertRaises(ValueError):
            ss._partition()

    def test_partition_05(self):
        np.random.seed(42)
        matrix = np.random.uniform(-1, 1, 9).reshape(3, 3)
        ss = ActiveSubspaces(dim=4)
        ss.evects = matrix
        with self.assertRaises(ValueError):
            ss._partition()

    def test_bootstrap_replicate_01(self):
        np.random.seed(42)
        matrix = np.random.uniform(-1, 1, 9).reshape(3, 3)
        weights = np.ones((3, 1)) / 3
        ss = ActiveSubspaces(dim=1)
        wei = ss._bootstrap_replicate(matrix, weights)[1]
        np.testing.assert_array_almost_equal(weights, wei)

    def test_bootstrap_replicate_02(self):
        np.random.seed(42)
        matrix = np.random.uniform(-1, 1, 9).reshape(3, 3)
        weights = np.ones((3, 1)) / 3
        ss = ActiveSubspaces(dim=1)
        mat = ss._bootstrap_replicate(matrix, weights)[0]
        true_matrix = np.array([[-0.88383278, 0.73235229, 0.20223002],
                                [0.19731697, -0.68796272, -0.68801096],
                                [-0.25091976, 0.90142861, 0.46398788]])
        np.testing.assert_array_almost_equal(true_matrix, mat)

    def test_bootstrap_replicate_03(self):
        np.random.seed(42)
        matrix = np.random.uniform(-1, 1, 27).reshape(3, 3, 3)
        weights = np.ones((3, 1)) / 3
        ss = ActiveSubspaces(dim=1)
        wei = ss._bootstrap_replicate(matrix, weights)[1]
        np.testing.assert_array_almost_equal(weights, wei)

    def test_bootstrap_replicate_04(self):
        np.random.seed(42)
        matrix = np.random.uniform(-1, 1, 27).reshape(3, 3, 3)
        weights = np.ones((3, 1)) / 3
        ss = ActiveSubspaces(dim=1)
        mat = ss._bootstrap_replicate(matrix, weights)[0]
        true_matrix = np.array([[[-0.13610996, -0.41754172, 0.22370579],
                                 [-0.72101228, -0.4157107, -0.26727631],
                                 [-0.08786003, 0.57035192, -0.60065244]],
                                [[-0.25091976, 0.90142861, 0.46398788],
                                 [0.19731697, -0.68796272, -0.68801096],
                                 [-0.88383278, 0.73235229, 0.20223002]],
                                [[-0.13610996, -0.41754172, 0.22370579],
                                 [-0.72101228, -0.4157107, -0.26727631],
                                 [-0.08786003, 0.57035192, -0.60065244]]])
        np.testing.assert_array_almost_equal(true_matrix, mat)

    def test_fit_bootstrap_ranges_01(self):
        np.random.seed(42)
        gradients = np.random.uniform(-1, 1, 60).reshape(30, 2)
        weights = np.ones((30, 1)) / 30
        ss = ActiveSubspaces(dim=1, n_boot=100)
        ss.fit(gradients=gradients, weights=weights)
        true_bounds_evals = np.array([[0.3000497, 0.59008536],
                                      [0.17398718, 0.40959827]])
        np.testing.assert_array_almost_equal(true_bounds_evals, ss.evals_br)

    def test_fit_bootstrap_ranges_02(self):
        np.random.seed(42)
        gradients = np.random.uniform(-1, 1, 60).reshape(30, 2)
        weights = np.ones((30, 1)) / 30
        ss = ActiveSubspaces(dim=1, method='exact', n_boot=100)
        ss.fit(gradients=gradients, weights=weights)
        true_bounds_subspace = np.array([[0.002618, 0.290506, 0.648893]])
        np.testing.assert_array_almost_equal(true_bounds_subspace, ss.subs_br)

    def test_fit_bootstrap_ranges_03(self):
        np.random.seed(42)
        gradients = np.random.uniform(-1, 1, 180).reshape(30, 3, 2)
        weights = np.ones((30, 1)) / 30
        ss = ActiveSubspaces(dim=1, n_boot=100)
        ss.fit(gradients=gradients, weights=weights)
        true_bounds_evals = np.array([[0.99330673, 1.62694823],
                                      [0.65987633, 1.11751475]])
        np.testing.assert_array_almost_equal(true_bounds_evals, ss.evals_br)

    def test_fit_bootstrap_ranges_04(self):
        np.random.seed(42)
        gradients = np.random.uniform(-1, 1, 180).reshape(30, 3, 2)
        weights = np.ones((30, 1)) / 30
        ss = ActiveSubspaces(dim=1, n_boot=100)
        ss.fit(gradients=gradients, weights=weights)
        true_bounds_subspace = np.array([[0.00109331, 0.30254992, 0.90447872]])
        np.testing.assert_array_almost_equal(true_bounds_subspace, ss.subs_br)

    def test_plot_eigenvalues_01(self):
        ss = ActiveSubspaces(dim=1)
        with self.assertRaises(TypeError):
            ss.plot_eigenvalues(figsize=(7, 7), title='Eigenvalues')

    def test_plot_eigenvalues_02(self):
        np.random.seed(42)
        gradients = np.random.uniform(-1, 1, 200).reshape(50, 4)
        weights = np.ones((50, 1)) / 50
        ss = ActiveSubspaces(dim=1, n_boot=200)
        ss.fit(gradients=gradients, weights=weights)
        with assert_plot_figures_added():
            ss.plot_eigenvalues(figsize=(7, 7), title='Eigenvalues')

    def test_plot_eigenvalues_03(self):
        np.random.seed(42)
        gradients = np.random.uniform(-1, 1, 200).reshape(50, 4)
        weights = np.ones((50, 1)) / 50
        ss = ActiveSubspaces(dim=1, n_boot=200)
        ss.fit(gradients=gradients, weights=weights)
        with assert_plot_figures_added():
            ss.plot_eigenvalues(n_evals=3, figsize=(7, 7), title='Eigenvalues')

    def test_plot_eigenvalues_04(self):
        np.random.seed(42)
        gradients = np.random.uniform(-1, 1, 200).reshape(50, 4)
        weights = np.ones((50, 1)) / 50
        ss = ActiveSubspaces(dim=1, n_boot=200)
        ss.fit(gradients=gradients, weights=weights)
        with self.assertRaises(TypeError):
            ss.plot_eigenvalues(n_evals=5, figsize=(7, 7))

    def test_plot_eigenvalues_05(self):
        np.random.seed(42)
        grad = np.array(
            [[-0.50183952, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [-1.26638196, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0.43017941, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0.65008914, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        gradients = (grad[i, :] for i in range(4))
        ss = ActiveSubspaces(dim=2, method='exact', n_boot=200)
        ss.fit(gradients=gradients)
        with assert_plot_figures_added():
            ss.plot_eigenvalues(figsize=(7, 7), title='Eigenvalues')

    def test_plot_eigenvectors_01(self):
        ss = ActiveSubspaces(dim=1)
        with self.assertRaises(TypeError):
            ss.plot_eigenvectors(figsize=(7, 7), title='Eigenvalues')

    def test_plot_eigenvectors_02(self):
        np.random.seed(42)
        gradients = np.random.uniform(-1, 1, 200).reshape(50, 4)
        weights = np.ones((50, 1)) / 50
        ss = ActiveSubspaces(dim=1, n_boot=200)
        ss.fit(gradients=gradients, weights=weights)
        with assert_plot_figures_added():
            ss.plot_eigenvectors(figsize=(7, 7), title='Eigenvectors')

    def test_plot_eigenvectors_03(self):
        np.random.seed(42)
        gradients = np.random.uniform(-1, 1, 200).reshape(50, 4)
        weights = np.ones((50, 1)) / 50
        ss = ActiveSubspaces(dim=1, n_boot=200)
        ss.fit(gradients=gradients, weights=weights)
        with assert_plot_figures_added():
            ss.plot_eigenvectors(n_evects=2,
                                 labels=[r'$x$', r'$y$', r'$r$', r'$z$'])

    def test_plot_eigenvectors_04(self):
        np.random.seed(42)
        gradients = np.random.uniform(-1, 1, 200).reshape(50, 4)
        weights = np.ones((50, 1)) / 50
        ss = ActiveSubspaces(dim=1, n_boot=200)
        ss.fit(gradients=gradients, weights=weights)
        with self.assertRaises(ValueError):
            ss.plot_eigenvectors(n_evects=10, figsize=(7, 7))

    def test_plot_eigenvectors_05(self):
        np.random.seed(42)
        grad = np.array(
            [[-0.50183952, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [-1.26638196, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0.43017941, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0.65008914, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        gradients = (grad[i, :] for i in range(4))
        ss = ActiveSubspaces(dim=2, method='exact', n_boot=200)
        ss.fit(gradients=gradients)
        with assert_plot_figures_added():
            ss.plot_eigenvectors(figsize=(7, 7), title='Eigenvectors')

    def test_plot_sufficient_summary_01(self):
        ss = ActiveSubspaces(dim=1)
        with self.assertRaises(TypeError):
            ss.plot_sufficient_summary(10, 10)

    def test_plot_sufficient_summary_02(self):
        np.random.seed(42)
        gradients = np.random.uniform(-1, 1, 200).reshape(50, 4)
        weights = np.ones((50, 1)) / 50
        ss = ActiveSubspaces(dim=3, n_boot=200)
        ss.fit(gradients=gradients, weights=weights)
        with self.assertRaises(ValueError):
            ss.plot_sufficient_summary(10, 10)

    def test_plot_sufficient_summary_03(self):
        np.random.seed(42)
        gradients = np.random.uniform(-1, 1, 200).reshape(50, 4)
        weights = np.ones((50, 1)) / 50
        ss = ActiveSubspaces(dim=2, n_boot=200)
        ss.fit(gradients=gradients, weights=weights)
        with assert_plot_figures_added():
            ss.plot_sufficient_summary(
                np.random.uniform(-1, 1, 100).reshape(25, 4),
                np.random.uniform(-1, 1, 25).reshape(-1, 1))

    def test_plot_sufficient_summary_04(self):
        np.random.seed(42)
        gradients = np.random.uniform(-1, 1, 200).reshape(50, 4)
        ss = ActiveSubspaces(dim=1, n_boot=100)
        ss.fit(gradients=gradients)
        with assert_plot_figures_added():
            ss.plot_sufficient_summary(
                np.random.uniform(-1, 1, 100).reshape(25, 4),
                np.random.uniform(-1, 1, 25).reshape(-1, 1))

    def test_frequent_directions_01(self):
        np.random.seed(42)
        grad = np.array(
            [[-0.50183952, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [-1.26638196, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0.43017941, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0.65008914, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        gradients = (grad[i, :] for i in range(4))
        ss = ActiveSubspaces(dim=2, method='exact', n_boot=150)
        evals, v = ss._frequent_directions(gradients=gradients)
        self.assertEqual(v.shape, (15, 2))
        self.assertEqual(evals.shape, (2, ))
