#!/usr/bin/env python
from os.path import join

from setuptools import setup, find_packages

MODULE_NAME = 'fklearn'         # package name used to install via pip (as shown in `pip freeze` or `conda list`)
MODULE_NAME_IMPORT = 'fklearn'  # this is how this module is imported in Python (name of the folder inside `src`)
REPO_NAME = 'fklearn'        # repository name


def requirements_from_pip(filename='requirements.txt'):
    with open(filename, 'r') as pip:
        return [l.strip() for l in pip if not l.startswith('#') and l.strip()]

core_deps = requirements_from_pip()
demos_deps = requirements_from_pip("requirements_demos.txt")
test_deps = requirements_from_pip("requirements_test.txt")

tools_deps = requirements_from_pip("requirements_tools.txt")

lgbm_deps = requirements_from_pip("requirements_lgbm.txt")
xgboost_deps = requirements_from_pip("requirements_xgboost.txt")
catboost_deps = requirements_from_pip("requirements_catboost.txt")

all_models_deps = lgbm_deps + xgboost_deps + catboost_deps
all_deps = all_models_deps + tools_deps
devel_deps = test_deps + all_deps

setup(name=MODULE_NAME,
      description="Functional machine learning",
      url='https://github.com/nubank/{:s}'.format(REPO_NAME),
      author="Nubank",
      package_dir={'': 'src'},
      packages=find_packages('src'),
      version=(open(join('src', MODULE_NAME, 'resources', 'VERSION'))
               .read().strip()),
      install_requires=core_deps,
      extras_require={"test_deps": test_deps,
                      "lgbm": lgbm_deps,
                      "xgboost": xgboost_deps,
                      "catboost": catboost_deps,
                      "tools": tools_deps,
                      "devel": devel_deps,
                      "all_models": all_models_deps,
                      "devel": devel_deps,
                      "all": all_deps},
      include_package_data=True,
      zip_safe=False,
      classifiers=['Programming Language :: Python :: 3.6'])

