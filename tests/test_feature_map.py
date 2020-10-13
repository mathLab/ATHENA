from unittest import TestCase
import numpy as np
from athena import FeatureMap, rff_map, rff_jac


class TestProjectionFactory(TestCase):
    def test_init_distr(self):
        fm = FeatureMap(distr='beta',
                        bias=None,
                        input_dim=3,
                        n_features=5,
                        params=[0.1, 0.3],
                        sigma_f=0.5)
        self.assertEqual(3, fm.input_dim)

    def test_init_bias(self):
        fm = FeatureMap(distr='cauchy',
                        bias=None,
                        input_dim=3,
                        n_features=7,
                        params=[0.1, 0.3],
                        sigma_f=1.5)
        self.assertIsNone(fm.bias)

    def test_init_input_dim(self):
        fm = FeatureMap(distr='cauchy',
                        bias=None,
                        input_dim=3,
                        n_features=7,
                        params=[0.1, 0.3],
                        sigma_f=1.5)
        self.assertEqual(3, fm.input_dim)

    def test_init_n_features(self):
        fm = FeatureMap(distr='dirichlet',
                        bias=None,
                        input_dim=3,
                        n_features=5,
                        params=[0.1, 0.3],
                        sigma_f=2.5)
        self.assertEqual(5, fm.n_features)

    def test_init_params(self):
        fm = FeatureMap(distr='laplace',
                        bias=None,
                        input_dim=2,
                        n_features=4,
                        params=[1.1, 2.3],
                        sigma_f=0.1)
        self.assertEqual([1.1, 2.3], fm.params)

    def test_init_sigma_f(self):
        fm = FeatureMap(distr='multivariate_normal',
                        bias=None,
                        input_dim=3,
                        n_features=5,
                        params=[0.1, 0.3],
                        sigma_f=0.1)
        self.assertEqual(0.1, fm.sigma_f)

    def test_init_fmap(self):
        fm = FeatureMap(distr='normal',
                        bias=None,
                        input_dim=5,
                        n_features=9,
                        params=[3.1, 0.3],
                        sigma_f=0.9)
        self.assertIs(rff_map, fm.fmap)

    def test_init_fmap_jac(self):
        fm = FeatureMap(distr='normal',
                        bias=None,
                        input_dim=5,
                        n_features=9,
                        params=[3.1, 0.3],
                        sigma_f=0.9)
        self.assertIs(rff_jac, fm.fmap_jac)

    def test_init__pr_matrix(self):
        fm = FeatureMap(distr='uniform',
                        bias=None,
                        input_dim=4,
                        n_features=10,
                        params=[0.6, 0.4],
                        sigma_f=0.2)
        self.assertIsNone(fm._pr_matrix)

    def test__compute_pr_matrix(self):
        np.random.seed(42)
        fm = FeatureMap(distr='uniform',
                        bias=None,
                        input_dim=2,
                        n_features=3,
                        params=[1.1, 1.2],
                        sigma_f=0.2)
        pr_matrix = fm._compute_pr_matrix()
        true_value = np.array([[1.137454, 1.195071], [1.173199, 1.159866],
                               [1.115602, 1.115599]])
        np.testing.assert_array_almost_equal(true_value, pr_matrix)

    def test_compute_fmap(self):
        np.random.seed(42)
        fm = FeatureMap(distr='laplace',
                        bias=np.ones((1, 3)),
                        input_dim=2,
                        n_features=3,
                        params=[0.8, 2.3],
                        sigma_f=0.9)
        inputs = np.random.uniform(size=(5, 2))
        fmap = fm.compute_fmap(inputs=inputs)
        true_value = np.array([[0.70465845, 0.44404558, 0.55780791],
                               [0.87857849, -0.95074097, 0.54437667],
                               [0.45910944, 0.24313274, 0.89833969],
                               [0.84035852, 1.02257209, 0.95970943],
                               [-0.62313605, -0.58107188, 0.57497268]])
        np.testing.assert_array_almost_equal(true_value, fmap)

    def test_compute_fmap_jac(self):
        np.random.seed(42)
        fm = FeatureMap(distr='laplace',
                        bias=np.ones((1, 3)),
                        input_dim=2,
                        n_features=3,
                        params=[0.8, 2.3],
                        sigma_f=0.9)
        inputs = np.random.uniform(size=(5, 2))
        fmap_jac = fm.compute_fmap_jac(inputs=inputs)
        true_value = np.array([[[-3.53084308, 3.923392],
                                [-2.20214773, 0.77718418],
                                [-0.94652067, -0.93418587]],
                               [[2.56578301, -2.85103934],
                                [-0.9835101, 0.34710137],
                                [-0.95558954, -0.94313656]],
                               [[4.30961695, -4.7887477],
                                [-2.36809012, 0.83574874],
                                [0.56400016, 0.55665027]],
                               [[2.82613179, -3.14033295],
                                [-0.43435979, 0.15329469],
                                [-0.43037806, -0.42476949]],
                               [[3.84444395, -4.2718581],
                                [-2.01936794, 0.71267736],
                                [-0.93447546, -0.92229763]]])
        np.testing.assert_array_almost_equal(true_value, fmap_jac)

    def test_tune_pr_matrix_none_01(self):
        np.random.seed(42)
        fm = FeatureMap(distr='multivariate_normal',
                        bias=None,
                        input_dim=2,
                        n_features=5,
                        params=[0.1, 0.3],
                        sigma_f=0.1)
        func = lambda x: np.sin(x[0] + x[1])
        bounds = (slice(-np.pi, 0, 0.25), slice(1, 2, 0.25))
        fm.tune_pr_matrix(func, bounds, args=(), method=None)
        true_value = np.array([[-0.88040291, -0.16933849],
                               [-1.14799804, 1.86532301],
                               [0.41502605, -0.28675804],
                               [-2.79908184, 0.93991175],
                               [0.83212168, 0.66449763]])
        np.testing.assert_array_almost_equal(true_value, fm.pr_matrix)

    def test_tune_pr_matrix_none_02(self):
        np.random.seed(42)
        fm = FeatureMap(distr='multivariate_normal',
                        bias=None,
                        input_dim=3,
                        n_features=5,
                        params=[0.1, 0.3, 0.1],
                        sigma_f=0.1)
        func = lambda x: np.sin(x[0] + x[1] + x[2])
        bounds = (slice(-np.pi, 0, 0.25), slice(1, 2, 0.25), slice(3, 4, 0.25))
        fm.tune_pr_matrix(func, bounds, args=(), method=None)
        true_value = np.array([[-0.61157462, -0.18290648, 0.96188282],
                               [0.22108191, -0.3097558, 2.94933463],
                               [0.44329736, 1.01522072, 3.05813247],
                               [0.43976152, -0.61304398, 1.05066301],
                               [1.62873959, -2.53103186, 0.46855792]])
        np.testing.assert_array_almost_equal(true_value, fm.pr_matrix)

    def test_tune_pr_matrix_none_03(self):
        np.random.seed(42)
        fm = FeatureMap(distr='multivariate_normal',
                        bias=None,
                        input_dim=4,
                        n_features=5,
                        params=[0.1, 0.3, 0.1, 0.3],
                        sigma_f=0.1)
        func = lambda x: np.sin(x[0] + x[1] + x[2] + x[3])
        bounds = (slice(-np.pi, 0, 0.25), slice(1, 2, 0.25), slice(3, 4, 0.25),
                  slice(3, 4, 0.25))
        fm.tune_pr_matrix(func, bounds, args=(), method=None)
        true_value = np.array(
            [[1.79243416, 0.96579734, 1.49638769, -1.17085025],
             [1.92976509, -0.97102892, 0.9688167, 0.30584914],
             [-0.76036359, -1.4831584, -1.66068389, 1.06297975],
             [-0.90661282, 0.67872439, -2.15831277, 1.50962728],
             [-0.42752186, -0.58457006, 1.01920618, -1.24015295]])
        np.testing.assert_array_almost_equal(true_value, fm.pr_matrix)

    def test_tune_pr_matrix_brute(self):
        np.random.seed(42)
        fm = FeatureMap(distr='uniform',
                        bias=None,
                        input_dim=2,
                        n_features=5,
                        params=[0.1, 0.3],
                        sigma_f=0.1)
        func = lambda x: np.sin(x[0] + x[1])
        bounds = (slice(-np.pi, 0, 0.25), slice(1, 2, 0.25))
        fm.tune_pr_matrix(func, bounds, args=(), method='brute')
        true_value = np.array([[-1.40312999, 1.27123589],
                               [0.25602505, -0.36286383],
                               [-2.41741768, -2.41752963],
                               [-2.87199219, 0.87884418],
                               [-0.35146163, 0.14499182]])
        np.testing.assert_array_almost_equal(true_value, fm.pr_matrix)

    def test_tune_pr_matrix_dual_annealing(self):
        np.random.seed(42)
        fm = FeatureMap(distr='beta',
                        bias=None,
                        input_dim=2,
                        n_features=5,
                        params=[0.1, 0.3],
                        sigma_f=0.1)
        func = lambda x: np.sin(x[0] + x[1])
        bounds = (slice(3, 4, 0.25), slice(1, 2, 0.25))
        fm.tune_pr_matrix(func, bounds, args=(), method='dual_annealing')
        true_value = np.array([[0.96136567,
                                0.68442507], [0.27881283, 0.69151686],
                               [0.57626534, 0.6273798], [0.3170462, 0.8636379],
                               [0.97294359, 0.99518304]])
        np.testing.assert_array_almost_equal(true_value, fm.pr_matrix)
