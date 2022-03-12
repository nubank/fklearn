#!/bin/bash

set -e

cur_dir=$(dirname ${BASH_SOURCE[0]})
source $cur_dir/helpers.sh

/usr/bin/env python3 -m venv venv

activate_venv

venv/bin/python -m pip install -q --upgrade setuptools
venv/bin/python -m pip install -q --upgrade pip
