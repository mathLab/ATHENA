from unittest import TestCase
import numpy as np
from athena.utils import Normalizer
from athena.local import TopDownHierarchicalAS, KMeansAS, KMedoidsAS, plot_scores
from athena.local_classification import ClassifyAS
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

class TestLocalAS(TestCase):
    def test_init_local_AS(self):
        las = TopDownHierarchicalAS()
        self.assertIsNone(las.inputs)
        self.assertIsNone(las.outputs)
        self.assertIsNone(las.gradients)
        self.assertIsNone(las.inputs_test)
        self.assertIsNone(las.outputs_test)
        self.assertIsNone(las.inputs_val)
        self.assertIsNone(las.outputs_val)

        self.assertIsNone(las.outputs_dim)
        self.assertIsNone(las.method)
        self.assertIsNone(las.as_dim)
        self.assertIsNone(las.clustering)
        self.assertIsNone(las.labels)
        self.assertIsNone(las.unique_labels)
        self.assertIsNone(las.full_as)
        self.assertIsNone(las.full_gpr)
        self.assertIsNone(las.local_ass)
        self.assertIsNone(las.local_gprs)
        self.assertIsNone(las.max_clusters)
        self.assertIsNone(las.random_state)