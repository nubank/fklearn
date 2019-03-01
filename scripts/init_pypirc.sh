#!bin/bash

set -e

echo -e "[pypi]" >> ~/.pypirc
echo -e "username = $PYPI_USER" >> ~/.pypirc
echo -e "password = $PYPI_PASSWORD" >> ~/.pypirc
