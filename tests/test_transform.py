import datetime
import logging

import numpy as np
import pandas as pd
import pytest
import torch

from cerulean import dimensions, factor, transform


def mock_train_data_dataframe():
    return pd.DataFrame({
        "a": [0, 0, 1, 0, 1],
        "b": [0, 1, 2, 0, 1],
        "c": [1, 0, 3, 2, 1]
    })


def test_convert_df_to_od_torch():
    df = mock_train_data_dataframe()
    logging.info(f"Original dataframe: {df}")
    out = transform._df2od_torch(
        df,
        [["a", "b"], ["b", "c"], ["c", "a"]],
        [(2, 3), (3, 4), (4, 2)]
    )
    logging.info(f"Converted df to od: {out}")
    # check explicit values -- based on dataframe and passed dims,
    # first element of ab is first element possible, so should have
    # flattened index zero; likewise, third element of "bc" should 
    # be max allowable by dims since it's the last for each
    # variable placement -- so shoudld be 3 * 4 - 1 = 11
    assert torch.equal(
        out["ab"][0].type(torch.long),
        torch.tensor(0).type(torch.long)
    )
    assert torch.equal(
        out["bc"][2].type(torch.long),
        torch.tensor(11).type(torch.long)
    )


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


def test_logdiff_stationary():
    records = mock_price_dataframe()
    records = records.resample('us').ffill()
    logging.info(f"Original records: {records}")

    s, s_inv = transform.make_stationarizer("logdiff")
    stationary, centralized = transform.to_stationary(
        np.mean, s, records
    )
    logging.info(f"Stationarized records: {stationary}")

    nonstationary = s_inv(stationary, centralized)
    logging.info(f"Nonstationary records: {nonstationary}")
    assert pd.testing.assert_frame_equal(
        nonstationary,
        records.iloc[1:],
        check_exact=False,
    ) is None


def test_diff_stationary():
    records = mock_price_dataframe()
    records = records.resample('us').ffill()
    logging.info(f"Original records: {records}")

    s, s_inv = transform.make_stationarizer("diff")
    stationary, centralized = transform.to_stationary(
        np.mean, s, records
    )
    logging.info(f"Stationarized records: {stationary}")

    nonstationary = s_inv(stationary, centralized)
    logging.info(f"Nonstationary records: {nonstationary}")
    assert pd.testing.assert_frame_equal(
        nonstationary,
        records.iloc[1:],
        check_exact=False,
    ) is None


@pytest.mark.slow
@pytest.mark.training
def test_inference_with_stationary():
    records = mock_price_dataframe()
    records = records.resample('us').ffill()
    logging.info(f"Original records: {records}")

    s, s_inv = transform.make_stationarizer("diff")
    stationary, centralized = transform.to_stationary(
        np.mean, s, records
    )
    logging.info(f"Stationarized records: {stationary}")

    n_cutpoints = 11
    discrete_stationary, bins = transform.continuous_to_variable_level(
        stationary,
        n_cutpoints,
    )

    discrete_stationary_minmax, minmax_bins = transform.continuous_to_variable_level(
        stationary,
        n_cutpoints,
        -0.02,
        0.02
    )

    dim_factory = dimensions.DimensionsFactory("NYSE", "NASDAQ", "BATS")
    dim_factory("NYSE", n_cutpoints)
    dim_factory("NASDAQ", n_cutpoints)
    dim_factory("BATS", n_cutpoints)
    d_nn = dim_factory(("NYSE", "NASDAQ"))
    d_nb = dim_factory(("NASDAQ", "BATS"))
    d_bn = dim_factory(("BATS", "NYSE"))

    logging.info("Training with auto min/max")
    factor_graph_2, losses_from_training_2 = factor.DiscreteFactorGraph.learn(
        (d_nn, d_nb, d_bn),
        discrete_stationary.rename(columns=dim_factory.mapping())
    )

    logging.info("Training with user-set min/max")
    factor_graph_2, losses_from_training_2 = factor.DiscreteFactorGraph.learn(
        (d_nn, d_nb, d_bn),
        discrete_stationary_minmax.rename(columns=dim_factory.mapping())
    )


@pytest.mark.bins
def test_stationary_to_nonstationary_bins():
    data = np.random.randn(1000)
    num_bins = 25
    _, bins = np.histogram(data, bins=num_bins)
    logging.info(f"Generated {num_bins} from normal data:\n{bins}")

    # shifted normal distribution
    _, f_inv = transform.make_stationarizer("diff")
    shift_mean = 101.0
    shift_normal_bins = transform._stationary_to_nonstationary_bins(f_inv, shift_mean, bins)
    logging.info(f"Converted bins to shfited normal with mean {shift_mean}:\n{shift_normal_bins}")

    # log normal distribution
    _, f_inv = transform.make_stationarizer("logdiff")
    shift_lognormal_bins = transform._stationary_to_nonstationary_bins(f_inv, shift_mean, bins)
    logging.info(f"Converted bins to shfited lognormal with scale {shift_mean}:\n{shift_lognormal_bins}")


@pytest.mark.bins
def test_stationary_bins_in_dims():
    # make initial bins
    data = np.random.randn(1000)
    num_bins = 25
    _, bins = np.histogram(data, bins=num_bins)
    
    # make some variables
    factory = dimensions.DimensionsFactory("Var1", "Var2")
    factory("Var1", num_bins)
    factory("Var2", num_bins)

    # set the bins
    var1_dim = factory.get_variable("Var1")
    var1_dim.set_bins(bins)
    logging.info(f"Made variable dimension with bins: {var1_dim}")

    # make a transform
    _, f_inv = transform.make_stationarizer("logdiff")

    # take the bins somewhere else
    new_loc = 4.0
    var1_dim.transform_bins(f_inv, new_loc)
    logging.info(f"Shifted variable dimension's bins: {var1_dim}")

    # see where the values lie
    way_too_negative = -9.0
    way_too_positive = 18.0
    normalish = 3.95
    low_bin = var1_dim[way_too_negative]
    logging.info(f"Value {way_too_negative} maps to bin {low_bin}")
    high_bin = var1_dim[way_too_positive]
    logging.info(f"Value {way_too_positive} maps to bin {high_bin}")
    normalish_bin = var1_dim[normalish]
    logging.info(f"Value {way_too_positive} maps to bin {normalish_bin}")

@pytest.mark.bins
def test_different_binning_strategies():
    records = mock_price_dataframe()
    records = records.resample('us').ffill()
    logging.info(f"Original records: {records}")

    s, s_inv = transform.make_stationarizer("diff")
    stationary, centralized = transform.to_stationary(
        np.mean, s, records
    )
    logging.info(f"Stationarized records: {stationary}")

    n_cutpoints = 11

    # choosing bin edges automatically
    discrete_stationary, bins = transform.continuous_to_variable_level(
        stationary,
        n_cutpoints,
    )
    logging.info(f"Choosing bins automatically: {bins}")
    logging.info(f"Data looks like:\n{discrete_stationary}")

    # setting bin edges by fiat
    discrete_stationary, bins = transform.continuous_to_variable_level(
        stationary,
        n_cutpoints,
        the_min=-1.0,
        the_max=1.0,
    )
    logging.info(f"Setting bin edges: {bins}")
    logging.info(f"Data looks like:\n{discrete_stationary}")

    # setting entire bins
    bins = np.array([-1.0, -0.3, 0., 0.01, 0.02, 0.8])
    discrete_stationary, bins = transform.continuous_to_variable_level(
        stationary,
        bins=bins,
    )
    logging.info(f"Setting entire bins: {bins}")
    logging.info(f"Data looks like:\n{discrete_stationary}")
