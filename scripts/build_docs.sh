#!bin/bash

set -e

cur_dir=$(dirname ${BASH_SOURCE[0]})
source $cur_dir/helpers.sh

activate_venv

venv/bin/python3 -m pip install -r docs/requirements.txt

cd docs/ && make html
