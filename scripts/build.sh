#!bin/bash

set -e

cur_dir=$(dirname ${BASH_SOURCE[0]})
source $cur_dir/helpers.sh

activate_venv

venv/bin/python3 -m pip install wheel
venv/bin/python3 setup.py sdist
venv/bin/python3 setup.py bdist_wheel
