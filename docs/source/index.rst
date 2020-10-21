Welcome to ATHENA's documentation!
===================================================

ATHENA: Advanced Techniques for High dimensional parameter spaces to Enhance Numerical Analysis


Description
--------------------
ATHENA is a Python package for reduction in parameter spaces.


Installation
--------------------
ATHENA requires requires numpy, matplotlib, sphinx (for the documentation) and nose (for local test). The code is compatible with Python 3.6 and above. It can be installed directly from the source code.


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

We made some tutorial examples. 


References
--------------------
To implement the numerical methods present in this package we followed these works:

- Constantine. Active subspaces: Emerging ideas for dimension reduction in parameter studies. Volume 2 SIAM Spotlights, 2015.
- Constantine et al. Python Active-subspaces Utility Library, Journal of Open Source Software, 1(5), 79, 2016.



Indices and tables
--------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

