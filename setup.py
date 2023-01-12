from setuptools import setup

meta = {}
with open("athena/meta.py") as fp:
    exec(fp.read(), meta)

# Package meta-data.
IMPORTNAME = meta['__title__']
PIPNAME = meta['__packagename__']
DESCRIPTION = (
    "Advanced Techniques for High dimensional parameter spaces to Enhance "
    "Numerical Analysis"
)
URL = 'https://github.com/mathLab/ATHENA'
MAIL = meta['__mail__']
AUTHOR = meta['__author__']
VERSION = meta['__version__']
KEYWORDS = (
    "parameter-space-reduction active-subspaces kernel-active-subspaces "
    "model-reduction sensitivity-analysis nonlinear-level-set-learning"
)

REQUIRED = [
    'numpy', 'scipy', 'matplotlib', 'torch', 'GPyOpt', 'scikit-learn', 'scikit-learn-extra'
]

EXTRAS = {
    'docs': ['Sphinx>=1.4', 'sphinx_rtd_theme'],
    'formatting': ['yapf'],
    'tutorials': ['pyro', 'pyhmc'],
    'test': ['pytest', 'pytest-cov'],
}

LDESCRIPTION = (
  'ATHENA is a Python package for reduction of high dimensional '
  'parameter spaces in the context of numerical analysis. It allows '
  'the use of several dimensionality reduction techniques such as '
  'Active Subspaces (AS), Kernel-based Active Subspaces (KAS), and '
  'Nonlinear Level-set Learning (NLL).\n'
  '\n'
  'It is particularly suited for the study of parametric PDEs, for '
  'sensitivity analysis, and for the approximation of engineering '
  'quantities of interest. It can handle both scalar and vectorial '
  'high dimensional functions, making it a useful tool also to reduce '
  'the burden of computational intensive optimization tasks.'
)

setup(
    name=PIPNAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LDESCRIPTION,
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Mathematics'
    ],
    keywords=KEYWORDS,
    url=URL,
    author=AUTHOR,
    author_email=MAIL,
    license='MIT',
    packages=[IMPORTNAME],
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    zip_safe=False
)
