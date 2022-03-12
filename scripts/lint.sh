#!/bin/bash

set -e

cur_dir=$(dirname ${BASH_SOURCE[0]})
source $cur_dir/helpers.sh

activate_venv

venv/bin/python -m pip install -q flake8==3.8.4
venv/bin/python -m flake8 \
  --ignore=E731,W503 \
  --filename=*.py \
  --exclude=__init__.py \
  --show-source \
  --statistics \
  --max-line-length=120 \
  src/ tests/
