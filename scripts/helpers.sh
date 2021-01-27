#!bin/bash

set -e

activate_venv() {
  . venv/bin/activate
}

install_package() {
  activate_venv
  venv/bin/python3 -m pip install -qe .$1
}
