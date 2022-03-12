#!bin/bash

set -e

cur_dir=$(dirname ${BASH_SOURCE[0]})
source $cur_dir/helpers.sh

activate_venv

venv/bin/python -m pip install wheel
venv/bin/python setup.py sdist
venv/bin/python setup.py bdist_wheel
