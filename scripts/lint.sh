#!/bin/bash

set -e

cur_dir=$(dirname ${BASH_SOURCE[0]})
source $cur_dir/helpers.sh

activate_venv

venv/bin/python3 -m pip install -q flake8==3.8.0
venv/bin/python3 -m flake8 \
  --ignore=E731,W503 \
  --filename=*.py \
  --exclude=__init__.py \
  --show-source \
  --statistics \
  --max-line-length=120 \
  src/ tests/
