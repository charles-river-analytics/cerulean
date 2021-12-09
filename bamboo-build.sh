#!/usr/bin/env bash

source /opt/conda/bin/conda

conda init bash

activate proper-fm

python -m pip install -r requirements.txt

python -m pytest --cov=cerulean

deactivate
