from unittest import TestCase
import numpy as np
from athena import FeatureMap, rff_map, rff_jac
from athena.kas import KernelActiveSubspaces
from athena.utils import average_rrmse, CrossValidation
from athena.projection_factory import ProjectionFactory


class TestProjectionFactory(TestCase):
    def test_init_distr_01(self):
        fm = FeatureMap(distr='beta',
                        bias=None,
                        input_dim=3,
                        n_features=5,
                        params=[0.1, 0.3],
                        sigma_f=0.5)
        self.assertEqual(ProjectionFactory('beta'), fm.distr)

    def test_init_distr_02(self):
        proj = ProjectionFactory('beta')
        fm = FeatureMap(distr=proj,
                        bias=None,
                        input_dim=3,
                        n_features=5,
                        params=[0.1, 0.3],
                        sigma_f=0.5)
        self.assertEqual(proj, fm.distr)

    def test_init_distr_03(self):
        with self.assertRaises(TypeError):
            fm = FeatureMap(distr=34,
                            bias=None,
                            input_dim=3,
                            n_features=5,
                            params=[0.1, 0.3],
                            sigma_f=0.5)

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
        true_value = np.array([[-0.42149623, 0.57314349, 0.73325259],
                               [0.65251962, 0.2455743, 0.73290642],
                               [0.42591728, 0.37857758, 0.53838879],
                               [-0.6921959, 0.65932346, 0.71051842],
                               [0.3605913, 0.37159967, 0.73375339]])
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
        true_value = np.array([[[1.53620402, -1.35337581],
                                [-0.40223905, 0.31509425],
                                [-0.03915164, -0.03881686]],
                               [[0.86249731, -0.75984894],
                                [-0.60576495, 0.4745264],
                                [-0.04318837, -0.04281907]],
                               [[1.52824153, -1.34636096],
                                [-0.5508609, 0.43151728],
                                [0.4047367, 0.40127589]],
                               [[0.62961265, -0.55468058],
                                [-0.28380577, 0.22231945],
                                [0.15175034, 0.15045276]],
                               [[1.63406118, -1.4395867],
                                [-0.55448363, 0.43435515],
                                [-0.03243032, -0.03215301]]])
        np.testing.assert_array_almost_equal(true_value, fmap_jac)

    def test_brute(self):
        np.random.seed(42)
        inputs = np.random.uniform(-1, 1, 10).reshape(5, 2)
        outputs = np.random.uniform(0, 5, 10).reshape(5, 2)
        gradients = np.random.uniform(-1, 1, 20).reshape(5, 2, 2)
        fm = FeatureMap(distr='laplace',
                        bias=np.random.uniform(-1, 1, 3),
                        input_dim=2,
                        n_features=3,
                        params=np.array([5.34265038]),
                        sigma_f=outputs.var())
        ss = KernelActiveSubspaces(dim=1, feature_map=fm)
        csv = CrossValidation(inputs=inputs,
                              outputs=outputs,
                              gradients=gradients,
                              folds=2,
                              subspace=ss)
        best = fm.tune_pr_matrix(func=average_rrmse,
                                 bounds=[slice(-2, 1, 0.2) for _ in range(1)],
                                 fn_args={'csv': csv},
                                 maxiter=10,
                                 save_file=False)[1]
        true = np.array([[-0.781768, -1.871064],
                         [-0.545585, -1.13183],
                         [1.961803,  0.95774]])
        np.testing.assert_array_almost_equal(true, best)

    def test_dual_annealing(self):
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
        best = fm.tune_pr_matrix(func=average_rrmse,
                                 bounds=[slice(-2, 1, 0.2) for _ in range(1)],
                                 fn_args={'csv': csv},
                                 method='dual_annealing',
                                 maxiter=5,
                                 save_file=False)[1]
        true = np.array([[-18.205881, 0.130872], [-4.232873, 1.833328],
                         [-5.631037, 2.571455]])
        np.testing.assert_array_almost_equal(true, best)

    # TODO: remove GPyOpt dependency with Emukit
    # def test_bso(self):
    #     np.random.seed(42)
    #     inputs = np.random.uniform(-1, 1, 10).reshape(5, 2)
    #     outputs = np.random.uniform(0, 5, 10).reshape(5, 2)
    #     gradients = np.random.uniform(-1, 1, 20).reshape(5, 2, 2)
    #     fm = FeatureMap(distr='laplace',
    #                     bias=np.random.uniform(-1, 1, 3),
    #                     input_dim=2,
    #                     n_features=3,
    #                     params=np.zeros(1),
    #                     sigma_f=outputs.var())
    #     ss = KernelActiveSubspaces(dim=1, feature_map=fm)
    #     csv = CrossValidation(inputs=inputs,
    #                           outputs=outputs,
    #                           gradients=gradients,
    #                           folds=2,
    #                           subspace=ss)
    #     best = fm.tune_pr_matrix(func=average_rrmse,
    #                              bounds=[slice(-2, 1, 0.2) for _ in range(1)],
    #                              fn_args={'csv': csv},
    #                              method='bso',
    #                              maxiter=10,
    #                              save_file=False)[1]
    #     true = np.array([[14.9646475, 4.2713126], [11.28870881, 8.33313971],
    #                      [1.16475035, 9.92216877]])
    #     np.testing.assert_array_almost_equal(true, best)
