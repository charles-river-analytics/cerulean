
import datetime
import itertools
import logging

import cerulean
import pandas as pd
import numpy as np


logging.basicConfig(level=logging.INFO)


def mock_price_dataframe():
    return pd.DataFrame({
        "NYSE": [
            100.10,
            100.11,
            100.10,
            100.08
        ],
        "NASDAQ": [
            100.10,
            100.11,
            100.11,
            100.09
        ],
        "BATS": [
            100.09,
            100.11,
            100.10, 
            100.09
        ]
    }, index=[
            datetime.datetime(2021,11,11,7,0,4,100000),
            datetime.datetime(2021,11,11,7,0,4,100003),
            datetime.datetime(2021,11,11,7,0,4,100012),
            datetime.datetime(2021,11,11,7,0,4,100015)
    ])


def main():
    # Step 0: get a bunch of data (here we'll pretend it's trading prices)
    prices = mock_price_dataframe()
    # Step 0.5: resample the data at whatever time resolution you want. 
    # we'll choose microsecond. Make sure to ffill to avoid violating causality!
    prices = prices.resample('us').ffill()
    logging.info(f"Original prices:\n{prices}")
    # Step 1: make a stationarizing transform and its inverse, and compute the
    # stationarized version of the time series. 
    s, s_inv = cerulean.transform.make_stationarizer("logdiff")
    stationary, centralized = cerulean.transform.to_stationary(
        np.mean, s, prices
    )
    logging.info(f"Used {s} to make stationarized time series:\n{stationary}")
    # Step 2: discretize the stationary data. You can either let cerulean figure out the 
    # values to use as the max / min bin edges (not recommended) or you can do it yourself.
    inflation = 0.2
    the_min = stationary.values.flatten().min() * (1.0 + inflation)
    the_max = stationary.values.flatten().max() * (1.0 + inflation)
    # you also need to choose how many bins to have. Here we'll choose (arbitrarily) 25
    n_bins = 25
    n_cutpoints = n_bins + 1
    discrete_stationary = cerulean.transform.continuous_to_variable_level(
        stationary,
        n_cutpoints,
        the_max=the_max,
        the_min=the_min,
    )
    logging.info(
        f"Used min = {the_min}, max = {the_max}, n_cutpoints = {n_cutpoints} to get discrete:\n{discrete_stationary}"
    )
    # Step 3: okay, now that we set up our data, we have to build the factor graph! We can specify its
    # structure using Dimensions objects, then learn its parameters from the data. Here we want a factor
    # for each pair of nodes.
    dim_factory = cerulean.dimensions.DimensionsFactory(*discrete_stationary.columns)
    # Register the variable's dimensionality with the dimension factory
    for location in discrete_stationary.columns:
        dim_factory(location, n_cutpoints)
    # create the appropriate factor dimensions
    factor_dims = [
        dim_factory(loc_pair)
        for loc_pair in itertools.combinations(discrete_stationary.columns, 2)
    ]
    logging.info(f"Factor specifications:\n{factor_dims}")
    logging.info(f"Variable name mappings:\n{dim_factory.mapping()}")
    # actually create the factor graph, learning its parameters from data
    variable_mapping = dim_factory.mapping()
    factor_graph, losses_from_training = cerulean.factor.DiscreteFactorGraph.learn(
        factor_dims,
        discrete_stationary.rename(columns=variable_mapping)
    )
    logging.info(f"Learned a factor graph:\n{factor_graph}")
    # Step 4: run inference! This is the fun part -- figure things out.
    # Example: infer all marginal distributions (remember this is still of the stationarized data!)
    marginals = {
        name : factor_graph.query(variable_mapping[name])
        for name in discrete_stationary.columns
    }
    logging.info(f"Inferred marginal distributions:\n{marginals}")






if __name__ == "__main__":
    main()