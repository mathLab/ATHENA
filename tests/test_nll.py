from unittest import TestCase
import numpy as np
from athena import NonlinearLevelSet, ForwardNet, BackwardNet, Normalizer
import torch
import os
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


def read_data():
    data_in = np.loadtxt('tests/data/naca0012.txt', skiprows=1, delimiter=',')
    real_inputs = data_in[:, 1:19]
    n_params = real_inputs.shape[1]
    lb = -0.01 * np.ones(n_params)
    ub = 0.01 * np.ones(n_params)
    normalizer = Normalizer(lb=lb, ub=ub)
    # inputs in [-1, 1]
    inputs = normalizer.fit_transform(real_inputs)
    lift = data_in[:, 19]
    # gradients with respect to normalized inputs
    grad_lift = data_in[:, 21:39]
    return inputs, lift, grad_lift


inputs, lift, grad_lift = read_data()
inputs_torch = torch.as_tensor(inputs, dtype=torch.double)
grad_torch = torch.as_tensor(grad_lift, dtype=torch.double)


class TestNonlinearLevelSet(TestCase):
    def test_init_n_layers(self):
        nll = NonlinearLevelSet(n_layers=2,
                                active_dim=1,
                                lr=0.1,
                                epochs=100,
                                dh=0.25)
        self.assertEqual(nll.n_layers, 2)

    def test_init_active_dim(self):
        nll = NonlinearLevelSet(n_layers=2,
                                active_dim=1,
                                lr=0.1,
                                epochs=100,
                                dh=0.25)
        self.assertEqual(nll.active_dim, 1)

    def test_init_lr(self):
        nll = NonlinearLevelSet(n_layers=2,
                                active_dim=1,
                                lr=0.1,
                                epochs=100,
                                dh=0.25)
        self.assertEqual(nll.lr, 0.1)

    def test_init_epochs(self):
        nll = NonlinearLevelSet(n_layers=2,
                                active_dim=1,
                                lr=0.1,
                                epochs=100,
                                dh=0.25)
        self.assertEqual(nll.epochs, 100)

    def test_init_dh(self):
        nll = NonlinearLevelSet(n_layers=2,
                                active_dim=1,
                                lr=0.1,
                                epochs=100,
                                dh=0.25)
        self.assertEqual(nll.dh, 0.25)

    def test_init_forward(self):
        nll = NonlinearLevelSet(n_layers=2,
                                active_dim=1,
                                lr=0.1,
                                epochs=100,
                                dh=0.25)
        self.assertIsNone(nll.forward)

    def test_init_backward(self):
        nll = NonlinearLevelSet(n_layers=2,
                                active_dim=1,
                                lr=0.1,
                                epochs=100,
                                dh=0.25)
        self.assertIsNone(nll.backward)

    def test_init_loss_vec(self):
        nll = NonlinearLevelSet(n_layers=2,
                                active_dim=1,
                                lr=0.1,
                                epochs=100,
                                dh=0.25)
        self.assertEqual(nll.loss_vec, [])

    def test_train_01(self):
        nll = NonlinearLevelSet(n_layers=2, active_dim=1, lr=0.02, epochs=1)
        nll.train(inputs=inputs_torch, gradients=grad_torch, interactive=False)
        self.assertIsInstance(nll.forward, ForwardNet)

    def test_train_02(self):
        nll = NonlinearLevelSet(n_layers=2, active_dim=1, lr=0.02, epochs=1)
        nll.train(inputs=inputs_torch, gradients=grad_torch, interactive=False)
        self.assertIsInstance(nll.backward, BackwardNet)

    def test_train_03(self):
        nll = NonlinearLevelSet(n_layers=2, active_dim=1, lr=0.02, epochs=1)
        nll.train(inputs=inputs_torch, gradients=grad_torch, interactive=False)
        self.assertIs(len(nll.loss_vec), 1)

    def test_train_04(self):
        nll = NonlinearLevelSet(n_layers=2, active_dim=1, lr=0.02, epochs=1)
        with self.assertRaises(ValueError):
            nll.train(inputs=inputs_torch,
                      gradients=grad_torch,
                      interactive=True)

    def test_train_05(self):
        nll = NonlinearLevelSet(n_layers=2, active_dim=1, lr=0.02, epochs=1)
        with assert_plot_figures_added():
            nll.train(inputs=inputs_torch,
                      gradients=grad_torch,
                      outputs=lift,
                      interactive=True)

    def test_forward_n_params(self):
        nll = NonlinearLevelSet(n_layers=2, active_dim=1, lr=0.02, epochs=1)
        nll.train(inputs=inputs_torch, gradients=grad_torch, interactive=False)
        self.assertEqual(nll.forward.n_params, 9)

    def test_backward_n_params(self):
        nll = NonlinearLevelSet(n_layers=2, active_dim=1, lr=0.02, epochs=1)
        nll.train(inputs=inputs_torch, gradients=grad_torch, interactive=False)
        self.assertEqual(nll.backward.n_params, 9)

    def test_plot_sufficient_summary_01(self):
        nll = NonlinearLevelSet(n_layers=2, active_dim=1, lr=0.02, epochs=1)
        nll.train(inputs=inputs_torch, gradients=grad_torch, interactive=False)
        with assert_plot_figures_added():
            nll.plot_sufficient_summary(inputs=inputs_torch, outputs=lift)

    def test_plot_sufficient_summary_02(self):
        nll = NonlinearLevelSet(n_layers=2, active_dim=2, lr=0.02, epochs=1)
        nll.train(inputs=inputs_torch, gradients=grad_torch, interactive=False)
        with self.assertRaises(ValueError):
            nll.plot_sufficient_summary(inputs=inputs_torch, outputs=lift)

    def test_plot_loss(self):
        nll = NonlinearLevelSet(n_layers=2, active_dim=1, lr=0.02, epochs=2)
        nll.train(inputs=inputs_torch, gradients=grad_torch, interactive=False)
        with assert_plot_figures_added():
            nll.plot_loss()

    def test_save_forward(self):
        nll = NonlinearLevelSet(n_layers=2, active_dim=1, lr=0.02, epochs=1)
        nll.train(inputs=inputs_torch, gradients=grad_torch, interactive=False)
        outfilename = 'tests/data/saved_forward.pth'
        nll.save_forward(outfilename)
        self.assertTrue(os.path.exists(outfilename))
        self.addCleanup(os.remove, outfilename)

    def test_load_forward(self):
        nll = NonlinearLevelSet(n_layers=2, active_dim=1, lr=0.02, epochs=1)
        nll.load_forward(infile='tests/data/forward_test.pth', n_params=18)
        self.assertIsInstance(nll.forward, ForwardNet)

    def test_save_backward(self):
        nll = NonlinearLevelSet(n_layers=2, active_dim=1, lr=0.02, epochs=1)
        nll.train(inputs=inputs_torch, gradients=grad_torch, interactive=False)
        outfilename = 'tests/data/saved_backward.pth'
        nll.save_backward(outfilename)
        self.assertTrue(os.path.exists(outfilename))
        self.addCleanup(os.remove, outfilename)

    def test_load_backward(self):
        nll = NonlinearLevelSet(n_layers=2, active_dim=1, lr=0.02, epochs=1)
        nll.load_backward(infile='tests/data/backward_test.pth', n_params=18)
        self.assertIsInstance(nll.backward, BackwardNet)


class TestForwardNet(TestCase):
    def test_init_n_params(self):
        nll = ForwardNet(n_params=6, n_layers=2, dh=0.25, active_dim=1)
        self.assertEqual(nll.n_params, 3)

    def test_init_n_layers(self):
        nll = ForwardNet(n_params=6, n_layers=2, dh=0.25, active_dim=1)
        self.assertEqual(nll.n_layers, 2)

    def test_init_dh(self):
        nll = ForwardNet(n_params=6, n_layers=2, dh=0.20, active_dim=1)
        self.assertEqual(nll.dh, 0.20)

    def test_init_omega(self):
        nll = ForwardNet(n_params=6, n_layers=2, dh=0.25, active_dim=1)
        self.assertEqual(nll.omega, slice(1))


class TestBackwardNet(TestCase):
    def test_init_n_params(self):
        nll = BackwardNet(n_params=6, n_layers=2, dh=0.25)
        self.assertEqual(nll.n_params, 3)

    def test_init_n_layers(self):
        nll = BackwardNet(n_params=6, n_layers=2, dh=0.25)
        self.assertEqual(nll.n_layers, 2)

    def test_init_dh(self):
        nll = BackwardNet(n_params=6, n_layers=2, dh=0.20)
        self.assertEqual(nll.dh, 0.20)
