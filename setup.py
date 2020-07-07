from setuptools import setup, find_packages

def readme():
    """
    This function just return the content of README.md
    """
    with open('README.md') as f:
        return f.read()

setup(name='athena',
      version='0.0.1',
      description='Advanced Techniques for High dimensional parameter spaces to Enhance Numerical Analysis',
      long_description=readme(),
      classifiers=[
        'Development Status ::  3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.5',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Mathematics'
      ],
      keywords='dimension_reduction mathematics active_subspaces',
      url='https://github.com/mathLab/ATHENA',
      author='Marco Tezzele, Francesco Romor',
      author_email='marcotez@gmail.com, francesco.romor@gmail.com',
      license='MIT',
      packages=find_packages(),
      install_requires=[
            'numpy',
            'scipy',
            'matplotlib',
            'torch',
            'Sphinx>=1.4',
            'sphinx_rtd_theme',
            'yapf'
      ],
      test_suite='nose.collector',
      tests_require=['nose'],
      include_package_data=True,
      zip_safe=False)
