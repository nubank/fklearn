#!bin/bash

set -e

cur_dir=$(dirname ${BASH_SOURCE[0]})
source $cur_dir/helpers.sh

activate_venv

venv/bin/python -m pip install -q twine
twine upload $1/*
