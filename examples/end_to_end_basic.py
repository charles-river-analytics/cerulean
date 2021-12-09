
import datetime
import itertools
import logging
import pathlib

import cerulean
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch


logging.basicConfig(level=logging.INFO)
DEFAULT_OUTPATH = pathlib.Path("end_to_end_basic")
DEFAULT_OUTPATH.mkdir(parents=True, exist_ok=True)


def mock_price_dataframe():
    return pd.DataFrame({
        "NYSE": [
            100.10,
            100.11,
            100.10,
            100.09,
            100.08
        ],
        "NASDAQ": [
            100.10,
            100.11,
            100.11,
            100.10,
            100.09
        ],
        "BATS": [
            100.09,
            100.11,
            100.10, 
            100.10,
            100.09
        ]
    }, index=[
            datetime.datetime(2021,11,11,7,0,4,100000),
            datetime.datetime(2021,11,11,7,0,4,100003),
            datetime.datetime(2021,11,11,7,0,4,100012),
            datetime.datetime(2021,11,11,7,0,4,100013),
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
    the_min = -0.02
    the_max = 0.02
    # you also need to choose how many bins to have.
    n_bins = 10
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
    prior_predictive_marginals = {
        name : factor_graph.query(variable_mapping[name])
        for name in discrete_stationary.columns
    }
    logging.info(f"Inferred prior predictive marginal factors:\n{prior_predictive_marginals}")
    prior_predictive = {
        name: f.table for (name, f) in prior_predictive_marginals.items()
    }
    logging.info(f"Inferred prior predictive:\n{prior_predictive}")
    # Suppose now we observe a particular log difference in the BATS time series. Let's infer
    # the other marginals. Remember to snapshot the model when you condition it on data!
    new_graph = factor_graph.snapshot()
    new_graph.post_evidence(variable_mapping["BATS"], torch.tensor(4))
    posterior_predictive_marginals = {
        name : new_graph.query(variable_mapping[name])
        for name in ["NYSE", "NASDAQ"]
    }
    posterior_predictive = {
        name: f.table for (name, f) in posterior_predictive_marginals.items()
    }
    logging.info(f"Inferred posterior predictive:\n{posterior_predictive}")
    
    # oooh, look at the pretty pictures!
    fig, axes = plt.subplots(2, 1)
    ax, ax2 = axes.flatten()
    ax.grid("on")
    ax.set_axisbelow(True)
    ax2.grid("on")
    ax2.set_axisbelow(True)
    labels = np.arange(len(posterior_predictive["NYSE"]))
    width = 0.35

    ## NYSE
    nyse_before_bars = ax.bar(
        labels - width/2, prior_predictive["NYSE"], width,
        label="Prior predictive",
        hatch="//",
        edgecolor="black"
    )
    ax.bar_label(
        nyse_before_bars,
        padding=3,
        fmt="%.3f",
        rotation=90,
    )
    nyse_after_bars = ax.bar(
        labels + width/2, posterior_predictive["NYSE"], width,
        label="Posterior predictive",
        hatch="\\",
        edgecolor="black"
    )
    ax.bar_label(
        nyse_after_bars,
        padding=3,
        fmt="%.3f",
        rotation=90
    )
    ax.set_yscale("log")
    ax.set_xticks(labels,)
    ax.set_xlabel(f"Stationarized value (NYSE)", fontsize=15)
    ax.set_ylabel(f"Probability", fontsize=15)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.65),
          fancybox=True, shadow=True, ncol=2)

    ## NASDAQ
    nasdaq_before_bars = ax2.bar(
        labels - width/2, prior_predictive["NASDAQ"], width,
        label="Prior predictive",
        hatch="//",
        edgecolor="black"
    )
    ax2.bar_label(
        nasdaq_before_bars,
        padding=3,
        fmt="%.3f",
        rotation=90,
    )
    nasdaq_after_bars = ax2.bar(
        labels + width/2, posterior_predictive["NASDAQ"], width,
        label="Posterior predictive",
        hatch="\\",
        edgecolor="black"
    )
    ax2.bar_label(
        nasdaq_after_bars,
        padding=3,
        fmt="%.3f",
        rotation=90
    )
    ax2.set_yscale("log")
    ax2.set_xticks(labels,)
    ax2.set_xlabel(f"Stationarized value (NASDAQ)", fontsize=15)
    ax2.set_ylabel(f"Probability", fontsize=15)

    fig.tight_layout()
    plt.savefig(DEFAULT_OUTPATH / f"end-to-end-basic-marginal.png")
    plt.close()


if __name__ == "__main__":
    main()