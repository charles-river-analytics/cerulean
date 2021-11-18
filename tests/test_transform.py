import datetime
import logging

import numpy as np
import pandas as pd
import torch

from models import dimensions, factor, transform


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


def test_inference_with_stationary():
    records = mock_price_dataframe()
    records = records.resample('us').ffill()
    logging.info(f"Original records: {records}")

    s, s_inv = transform.make_stationarizer("diff")
    stationary, centralized = transform.to_stationary(
        np.mean, s, records
    )
    logging.info(f"Stationarized records: {stationary}")

    n_cutpoints = 111
    discrete_stationary = transform.continuous_to_variable_level(
        stationary,
        n_cutpoints,
    )

    discrete_stationary_minmax = transform.continuous_to_variable_level(
        stationary,
        n_cutpoints,
        -0.02,
        0.02
    )

    names = transform.get_names2strings("NYSE", "NASDAQ", "BATS")

    nyse_dim = dimensions.VariableDimensions(names["NYSE"], n_cutpoints)
    nasdaq_dim = dimensions.VariableDimensions(names["NASDAQ"], n_cutpoints)
    bats_dim = dimensions.VariableDimensions(names["BATS"], n_cutpoints)

    d_nn = dimensions.FactorDimensions(
        nyse_dim, nasdaq_dim
    )
    d_nb = dimensions.FactorDimensions(
        nasdaq_dim, bats_dim
    )
    d_bn = dimensions.FactorDimensions(
        bats_dim, nyse_dim
    )
    factor_graph, losses_from_training = factor.DiscreteFactorGraph.learn(
        (d_nn, d_nb, d_bn),
        discrete_stationary.rename(columns=names)
    )
