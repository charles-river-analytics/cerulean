import datetime
from typing import Callable, Iterable, Literal

import mypy
import numpy as np
import pandas as pd


StationaryTransform = Literal["diff", "logdiff"]


def make_stationarizer(
    method: StationaryTransform
) -> Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
    if method == "diff":
        f = lambda x, y: x[1:] - y[:-1]
    else:
        f = lambda x, y: 100 * (np.log(x[1:]) - np.log(y[:-1]))
    def stationarizer(
        records: pd.DataFrame,
        centralized_records: pd.DataFrame
    ) -> pd.DataFrame:
        df = pd.DataFrame()
        for col in records.columns:
            df[col] = f(
                records[col].values,
                centralized_records.centralized.values
            )
        df.index = records.index[1:]
        return df
    return stationarizer


def to_stationary(
    centralizer: Callable,
    stationarizer: Callable,
    records: pd.DataFrame
) -> pd.DataFrame:
    centralized_records = pd.DataFrame()
    centralized_records["centralized"] = records.apply(centralizer, axis=1)
    return stationarizer(records, centralized_records)
