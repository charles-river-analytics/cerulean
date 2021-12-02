# cerulean

`cerulean` is a library for learning, exact inference, and constraint satisfaction using discrete factor graphs. 
It is developed as part of the PROPER-FM effort under DARPA I2O ECoSystemic.

## Installation

To install and work on this package locally:
+ `conda create --name proper-fm python=3.9`
+ `/opt/anaconda3/envs/proper-fm/bin/python -m pip install -r requirements.txt`

To install as part of another project:
+ `cd your/cerulean/dir`
+ `git clone https://git.collab.cra.com/scm/prop/cerulean.git`
+ `pip install -e your/cerulean/dir`

## Testing
This project uses `pytest`. In the top-level directory, run `/opt/anaconda3/envs/proper-fm/bin/python -m pytest --cov=cerulean` 
to execute all tests and see statistics of test coverage.

## Documentation

This project uses Sphinx to build its documentation. To build the documentation:

+ `cd docs`
+ `make html` (to build documentation as html)
+ `make latexpdf` (to build documentation as pdf)

## Examples

In order to run the examples, you have to install `cerulean` as a python package. To do that, just navigate to the directory in which 
it's located and `/opt/anaconda3/envs/proper-fm/bin/python -m pip install -e .`.

## Acknowledgements and other information

We are grateful for funding from DARPA under contract HR00112290006.

Copyright Charles River Analytics Inc., David Rushing Dewhurst, Joseph Campolongo, and Mike Reposa, 2021 - present.
All rights reserved.
This library is *not* to be used outside Charles River Analytics or DARPA without authorization from DARPA PM or higher authority.
