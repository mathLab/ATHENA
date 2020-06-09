"""
Module for Kernel-based Active Subspaces.

Reference:
- Francesco Romor, Marco Tezzele, Andrea Lario, Gianluigi Rozza.
Kernel-based Active Subspaces with application to CFD problems using
Discontinuous Galerkin Method. 2020. 
arxiv: 
"""
import numpy as np
from .subspaces import Subspaces


class KernelActiveSubspaces(Subspaces):
    """Kernel Active Subspaces class
    
    """
    def __init__(self):
        super().__init__()

    def compute(self, inputs=None, outputs=None, gradients=None):
        pass

    def forward(self, inputs):
        pass
