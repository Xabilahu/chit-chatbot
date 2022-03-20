#!/bin/bash

python_version=$(python3 -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')

if [[ "${python_version}" != "3.6" ]]
then
    >&2 echo "[ERROR] Needed Python3 version is 3.6, found ${python_version}"
    exit 1
fi

virtualenv .venv
. .venv/bin/activate
pip3 install -r requirements.txt
pip3 install --upgrade pre-commit
pre-commit install
