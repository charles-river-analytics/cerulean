import datetime
import logging

import numpy as np
import pandas as pd

from models import transform


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


def test_integration():
    records = mock_price_dataframe()
    records = records.resample('us').ffill()
    logging.info(f"Original records: {records}")

    stationarizer = transform.make_stationarizer("logdiff")
    test = transform.to_stationary(
        np.mean, stationarizer, records
    )
    logging.info(f"Stationarized records: {test}")
