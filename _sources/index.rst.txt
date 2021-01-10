Welcome to ATHENA's documentation!
===================================================

.. image:: _static/logo_athena.png
   :height: 150px
   :width: 150 px
   :align: right

ATHENA: Advanced Techniques for High dimensional parameter spaces to Enhance Numerical Analysis


Description
--------------------
ATHENA is a Python package for reduction of high dimensional parameter spaces in the context of numerical analysis. It allows the use of several dimensionality reduction techniques such as Active Subspaces (AS), Kernel-based Active Subspaces (KAS), and Nonlinear Level-set Learning (NLL). It is particularly suited for the study of parametric PDEs, for sensitivity analysis, and for the approximation of engineering quantities of interest. It can handle both scalar and vectorial high dimensional functions, making it a useful tool also to reduce the burden of computational intensive optimization tasks. 


Installation
--------------------
ATHENA requires requires numpy, matplotlib, scipy, torch, GPy, GPyOpt, sphinx (for the documentation) and nose (for local test). The code is compatible with Python 3.6 and above. It can be installed directly from the source code available at the official GitHub `repository <https://github.com/mathLab/ATHENA>`_.


Installing from source
^^^^^^^^^^^^^^^^^^^^^^^^
To install the latest version of the package just type
::

    pip install git+https://github.com/mathLab/ATHENA.git

The official distribution is on GitHub, and you can clone the repository using
::

    git clone https://github.com/mathLab/ATHENA

To install the package just type:
::

    python setup.py install

To uninstall the package you have to rerun the installation and record the installed files in order to remove them:

::

    python setup.py install --record installed_files.txt
    cat installed_files.txt | xargs rm -rf


Developer's Guide
--------------------

.. toctree::
   :maxdepth: 1

   code
   tutorials
   contact
   contributing
   LICENSE


Tutorials
--------------------

We made several tutorials to present the main features of the package. Please refer to the :doc:`dedicated page <tutorials>` of the documentation.


References
--------------------
To implement the numerical methods present in this package we followed these works:

- Constantine. *Active subspaces: Emerging ideas for dimension reduction in parameter studies*. Volume 2 SIAM Spotlights, 2015. [`DOI <https://doi.org/10.1137/1.9781611973860>`_]
- Constantine et al. *Python Active-subspaces Utility Library*, Journal of Open Source Software, 1(5), 79, 2016. [`DOI <https://doi.org/10.21105/joss.00079>`_]
- Romor, Tezzele, Lario, Rozza. *Kernel-based Active Subspaces with application to CFD parametric problems using Discontinuous Galerkin method*. 2020. [`arXiv <https://arxiv.org/abs/2008.12083>`_]
- Zhang, Zhang, Hinkle. *Learning nonlinear level sets for dimensionality reduction in function approximation*. Advances in Neural Information Processing Systems 32: Annual Conference on Neural Information Processing Systems 2019, NeurIPS 2019, 8-14 December 2019, Vancouver, BC, Canada. [`arXiv <https://arxiv.org/abs/1902.10652>`_]


Indices and tables
--------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

