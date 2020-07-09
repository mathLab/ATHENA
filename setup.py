from setuptools import setup
import athena

# Package meta-data.
NAME = athena.__title__
DESCRIPTION = 'Advanced Techniques for High dimensional parameter spaces to ' \
              'Enhance Numerical Analysis'
URL = 'https://github.com/mathLab/ATHENA'
MAIL = athena.__mail__
AUTHOR = athena.__author__
VERSION = athena.__version__
KEYWORDS = 'parameter-space-reduction active-subspaces ' \
           'kernel-active-subspaces model-reduction sensitivity-analysis ' \
           'nonlinear-level-set-learning'

REQUIRED = ['numpy', 'scipy', 'matplotlib', 'torch']

EXTRAS = {
    'docs': ['Sphinx>=1.4', 'sphinx_rtd_theme'],
    'formatting': ['yapf'],
    'tutorials': ['pyro'],
}

LDESCRIPTION = ('ATHENA is a Python package for reduction in parameter spaces.')

setup(name=NAME,
      version=VERSION,
      description=DESCRIPTION,
      long_description=LDESCRIPTION,
      classifiers=[
          'Development Status ::  3 - Alpha',
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
      packages=[NAME],
      install_requires=REQUIRED,
      extras_require=EXTRAS,
      test_suite='nose.collector',
      tests_require=['nose'],
      include_package_data=True,
      zip_safe=False)
