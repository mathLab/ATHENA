
<p align="center">
    <a href="https://github.com/mathLab/ATHENA/blob/master/LICENSE" target="_blank">
        <img alt="Software License" src="https://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat-square">
    </a>
    <a href="https://travis-ci.org/mathLab/ATHENA" target="_blank">
        <img alt="Build Status" src="https://travis-ci.org/mathLab/ATHENA.svg">
    </a>
    <a href="https://coveralls.io/github/mathLab/ATHENA" target="_blank">
        <img alt="Coverage Status" src="https://coveralls.io/repos/github/mathLab/ATHENA/badge.svg">
    </a>
    <a href="https://www.codacy.com/manual/mathLab/ATHENA?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=mathLab/ATHENA&amp;utm_campaign=Badge_Grade" target="_blank">
        <img alt="Codacy Badge" src="https://api.codacy.com/project/badge/Grade/da9bf8c772a74a038f3b44a91748c91b">
    </a>
</p>

**ATHENA**: Advanced Techniques for High dimensional parameter spaces to Enhance Numerical Analysis

## Table of contents
* [Description](#description)
* [Dependencies and installation](#dependencies-and-installation)
	* [Installing from source](#installing-from-source)
* [Documentation](#documentation)
* [Testing](#testing)
* [Examples and Tutorials](#examples)
* [References](#references)
	* [Recent works with ATHENA](#recent-works-with-athena)
* [Authors and contributors](#authors-and-contributors)
* [How to contribute](#how-to-contribute)
	* [Submitting a patch](#submitting-a-patch) 
* [License](#license)

## Description
**ATHENA** is a Python package for reduction in parameter spaces. 


## Dependencies and installation
**ATHENA** requires requires `numpy`, `matplotlib`, `sphinx` (for the documentation) and `nose` (for local test). The code is compatible with Python 3.6 and above. It can be installed directly from the source code.


### Installing from source
To install the latest version of the package just type:
```bash
> pip install git+https://github.com/mathLab/ATHENA.git
```

The official distribution is on GitHub, and you can clone the repository using
```bash
> git clone https://github.com/mathLab/ATHENA
```

To install your own local branch you can use the `setup.py` file
```bash
> python setup.py install
```

To uninstall the package you have to rerun the installation and record the installed files in order to remove them:

```bash
> python setup.py install --record installed_files.txt
> cat installed_files.txt | xargs rm -rf
```

## Documentation
**ATHENA** uses [Sphinx](http://www.sphinx-doc.org/en/stable/) for code documentation. You can view the documentation online [here](http://mathlab.github.io/ATHENA/). To build the html version of the docs locally simply:

```bash
> cd docs
> make html
```

The generated html can be found in `docs/build/html`. Open up the `index.html` you find there to browse.


## Testing

We are using Travis CI for continuous intergration testing. You can check out the current status [here](https://travis-ci.org/mathLab/ATHENA).

To run tests locally (`nose` is required):

```bash
> python test.py
```

## Examples and Tutorials
You can find useful tutorials on how to use the package in the [tutorials](tutorials/README.md) folder.


## References
To implement the numerical methods present in this package we followed these works:
* Constantine. *Active subspaces: Emerging ideas for dimension reduction in parameter studies*. Volume 2 SIAM Spotlights, 2015. [[DOI](https://doi.org/10.1137/1.9781611973860)].
* Constantine et al. Python Active-subspaces Utility Library, Journal of Open Source Software, 1(5), 79, 2016. [[DOI](https://doi.org/10.21105/joss.00079)].


### Recent works with ATHENA
Here there is a list of the scientific works involving **ATHENA** you can consult and/or cite. If you want to add one, please open a PR.

## Authors and contributors
**ATHENA** is currently developed and mantained at [SISSA mathLab](http://mathlab.sissa.it/) by
* [Marco Tezzele](mailto:marcotez@gmail.com)
* [Francesco Romor](mailto:francesco.romor@gmail.com)

under the supervision of [Prof. Gianluigi Rozza](mailto:gianluigi.rozza@sissa.it).

Contact us by email for further information or questions about **ATHENA**, or suggest pull requests. Contributions improving either the code or the documentation are welcome!


## How to contribute
We'd love to accept your patches and contributions to this project. There are just a few small guidelines you need to follow.

### Submitting a patch

  1. It's generally best to start by opening a new issue describing the bug or
     feature you're intending to fix.  Even if you think it's relatively minor,
     it's helpful to know what people are working on.  Mention in the initial
     issue that you are planning to work on that bug or feature so that it can
     be assigned to you.

  2. Follow the normal process of [forking][] the project, and setup a new
     branch to work in.  It's important that each group of changes be done in
     separate branches in order to ensure that a pull request only includes the
     commits related to that bug or feature.

  3. To ensure properly formatted code, please make sure to use 4
     spaces to indent the code. The easy way is to run on your bash the provided
     script: ./code_formatter.sh. You should also run [pylint][] over your code.
     It's not strictly necessary that your code be completely "lint-free",
     but this will help you find common style issues.

  4. Any significant changes should almost always be accompanied by tests.  The
     project already has good test coverage, so look at some of the existing
     tests if you're unsure how to go about it. We're using [coveralls][] that
     is an invaluable tools for seeing which parts of your code aren't being
     exercised by your tests.

  5. Do your best to have [well-formed commit messages][] for each change.
     This provides consistency throughout the project, and ensures that commit
     messages are able to be formatted properly by various git tools.

  6. Finally, push the commits to your fork and submit a [pull request][]. Please,
     remember to rebase properly in order to maintain a clean, linear git history.

[forking]: https://help.github.com/articles/fork-a-repo
[pylint]: https://www.pylint.org/
[coveralls]: https://coveralls.io
[well-formed commit messages]: http://tbaggery.com/2008/04/19/a-note-about-git-commit-messages.html
[pull request]: https://help.github.com/articles/creating-a-pull-request


## License

See the [LICENSE](LICENSE.rst) file for license rights and limitations (MIT).
