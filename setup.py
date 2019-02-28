#!/usr/bin/env python
from os.path import join

from setuptools import setup, find_packages

MODULE_NAME = 'fklearn'         # package name used to install via pip (as shown in `pip freeze` or `conda list`)
MODULE_NAME_IMPORT = 'fklearn'  # this is how this module is imported in Python (name of the folder inside `src`)
REPO_NAME = 'fklearn'        # repository name


def requirements_from_pip(filename='requirements.txt'):
    with open(filename, 'r') as pip:
        return [l.strip() for l in pip if not l.startswith('#') and l.strip()]


setup(name=MODULE_NAME,
      description="Functional machine learning",
      url='https://github.com/nubank/{:s}'.format(REPO_NAME),
      author="Nubank",
      package_dir={'': 'src'},
      packages=find_packages('src'),
      version=(open(join('src', MODULE_NAME, 'resources', 'VERSION'))
               .read().strip()),
      install_requires=requirements_from_pip(),
      extras_require={"test_deps": requirements_from_pip('requirements_test.txt')},
      include_package_data=True,
      zip_safe=False,
      classifiers=['Programming Language :: Python :: 3.6'])

