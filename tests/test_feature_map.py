from unittest import TestCase
from athena import FeatureMap, rff_map, rff_jac
import numpy as np


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