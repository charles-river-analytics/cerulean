# PROPER-FM models

## Installation

+ `conda create --name proper-fm python=3.9`
+ `/opt/anaconda3/envs/proper-fm/bin/python -m pip install -r requirements.txt`

## Testing
This project uses `pytest`. 

+  In the top-level directory, run `/opt/anaconda3/envs/proper-fm/bin/python -m pytest --cov=models` 
    to execute all tests and see statistics of test coverage.

## Documentation

This project uses Sphinx to build its documentation. To build the documentation:

+ `cd docs`
+ `make html` (to build documentation as html)
+ `make latexpdf` (to build documentation as pdf)