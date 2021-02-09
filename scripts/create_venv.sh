#!/bin/bash

set -e

cur_dir=$(dirname ${BASH_SOURCE[0]})
source $cur_dir/helpers.sh

/usr/local/bin/python3 -m venv venv

activate_venv

venv/bin/python3 -m pip install -q --upgrade setuptools
