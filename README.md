# cerulean

`cerulean` is a library for learning, exact inference, and constraint satisfaction using discrete factor graphs.
It is fast, using multiple complementary memoization strategies for maximal speed at the cost of higher memory usage, and has a convenient, software engineering-oriented interface. 
It builds on `pyro` for probabilistic modeling fundamentals and `opt-einsum` for contraction path computation.

## Examples

There are many examples in the [examples directory](./examples/):

+ [Random inference](./examples/random_inference.py): a collection of random inference problems for benchmarking. 
    Includes (a) inference of marginal densities in chain models; (b) inference of marginals and N-1 dimensional (all but one variable) in a 
    fully connected network.
+ [Random graph inference](./examples/benchmark.py): queries on random graphs (ER model) of varying number of nodes, connection probability, and variable dimensions.
+ [End-to-end basic](./examples/end_to_end_basic.py): example of an end-to-end inference problem using (hypothetical) quote data from three trading venues.
    Includes application of stationarizing transforms ([`cerulean.transform`](./cerulean/transform.py)).


## Installation

To install and work on this package locally:
+ `conda create --name cerulean python=3.9`
+ `/my/anaconda/envs/cerulean/bin/python -m pip install -r requirements.txt`

To install as part of another project:
+ `cd your/cerulean/dir`
+ `git clone https://github.com/charles-river-analytics/cerulean.git`
+ `cd my/other/project/dir`
+ `pip install -e your/cerulean/dir`
  
## Testing
This project uses `pytest`. In the top-level directory, run `/my/anaconda/envs/cerulean/bin/python -m pytest --cov=cerulean` 
to execute all tests and see statistics of test coverage.

## Documentation

This project uses Sphinx to build its documentation. To build the documentation:

+ `cd docs`
+ `make html` (to build documentation as html)
+ `make latexpdf` (to build documentation as pdf)

## Examples

In order to run the examples, you have to install `cerulean` as a python package. To do that, just navigate to the directory in which 
it's located and `/my/anaconda/envs/cerulean/bin/python -m pip install -e .`.

## Acknowledgements and other information

This work was funded by DARPA under contract HR00112290006.
Copyright Charles River Analytics Inc., 2021 - present. `cerulean` is released under the LGPLv3 license.
Approved for public release, distribution is unlimited.
