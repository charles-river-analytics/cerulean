# cerulean

`cerulean` is a library for learning, exact inference, and constraint satisfaction using factor graphs. 
It is developed as part of the PROPER-FM program.

## Installation

To install and work on this package locally:
+ `conda create --name proper-fm python=3.9`
+ `/opt/anaconda3/envs/proper-fm/bin/python -m pip install -r requirements.txt`

To install as part of another project:
+ `cd your/cerulean/dir`
+ `git clone https://git.collab.cra.com/scm/prop/models.git`
+ `pip install -e your/cerulean/dir`

## Testing
This project uses `pytest`. 

+  In the top-level directory, run `/opt/anaconda3/envs/proper-fm/bin/python -m pytest --cov=cerulean` 
    to execute all tests and see statistics of test coverage.

## Documentation

This project uses Sphinx to build its documentation. To build the documentation:

+ `cd docs`
+ `make html` (to build documentation as html)
+ `make latexpdf` (to build documentation as pdf)

## Acknowledgements and other information

We are grateful for funding from DARPA under the ECoSystemic program.

Copyright Charles River Analytics Inc., David Rushing Dewhurst, Joseph Campolongo, and Mike Reposa, 2021 - present.
All rights reserved.
This library is *not* to be used outside Charles River Analytics without authorization from DM VP or higher authority.
