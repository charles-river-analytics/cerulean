@echo off

rem Activate the conda environment
call activate proper-fm

rem Update cerulean requirements
python -m pip install -r requirements.txt

rem Run unit tests and coverage
python -m pytest --cov=cerulean

rem Deactivate the environment
call conda deactivate
