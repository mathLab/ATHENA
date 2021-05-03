from setuptools import setup

meta = {}
with open("athena/meta.py") as fp:
    exec(fp.read(), meta)

# Package meta-data.
IMPORTNAME = meta['__title__']
PIPNAME = meta['__packagename__']
DESCRIPTION = 'Advanced Techniques for High dimensional parameter spaces to ' \
              'Enhance Numerical Analysis'
URL = 'https://github.com/mathLab/ATHENA'
MAIL = meta['__mail__']
AUTHOR = meta['__author__']
VERSION = meta['__version__']
KEYWORDS = 'parameter-space-reduction active-subspaces ' \
           'kernel-active-subspaces model-reduction sensitivity-analysis ' \
           'nonlinear-level-set-learning'

REQUIRED = ['numpy', 'scipy', 'matplotlib', 'torch', 'GPy', 'GPyOpt', 'scikit-learn']

EXTRAS = {
    'docs': ['Sphinx>=1.4', 'sphinx_rtd_theme'],
    'formatting': ['yapf'],
    'tutorials': ['pyro', 'pyhmc'],
}

LDESCRIPTION = ('ATHENA is a Python package for reduction in parameter spaces.')

setup(name=PIPNAME,
      version=VERSION,
      description=DESCRIPTION,
      long_description=LDESCRIPTION,
      classifiers=[
          'Development Status :: 5 - Production/Stable',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3.6',
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
      test_suite='nose.collector',
      tests_require=['nose'],
      include_package_data=True,
      zip_safe=False)
