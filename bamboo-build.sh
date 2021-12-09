#!/usr/bin/env bash

source /opt/conda/bin/conda

conda init bash

rem Activate the conda environment
conda activate proper-fm

rem Update cerulean requirements
python -m pip install -r requirements.txt

rem Run unit tests and coverage
python -m pytest --cov=cerulean

rem Deactivate the environment
conda deactivate
