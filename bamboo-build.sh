#!/usr/bin/env bash

source /opt/conda/bin/conda

conda init bash

conda create -y --name proper-fm python=3.9

activate proper-fm

python -m pip install -r requirements.txt

python -m pytest --cov=cerulean

conda deactivate
