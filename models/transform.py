import datetime
from typing import Callable, Iterable, Literal

import mypy
import numpy as np
import pandas as pd


StationaryTransform = Literal["diff", "logdiff"]


def make_stationarizer(
    method: StationaryTransform
) -> tuple[Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame],...]:
    """
    Generates a (forward, partial inverse) transform pair that converts between
    a set of possibly-nonstationary time series and a likely-stationary set of
    time series. 

    Each transform takes two `pd.DataFrame` positional arguments. The forward transform
    takes the original records as its first argument and the centralized record as its
    second argument.

    + `method`: one of "diff" or "logdiff". If "diff", the forward method method will compute
        the first difference between the records and the centralized record. If "logdiff", the 
        forward method will compute 100 * the first log difference between the records and the
        centralized record. 
    """
    if method == "diff":
        f = lambda x, y: x[1:] - y[:-1]
        f_inv = lambda dx, y: y[:-1] + dx
    else:
        f = lambda x, y: 100.0 * (np.log(x[1:]) - np.log(y[:-1]))
        f_inv = lambda dx, y: np.exp(dx / 100.0 + np.log(y[:-1]))

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
    
    def inverse_stationarizer(
        dx: pd.DataFrame,
        centralized_records: pd.DataFrame
    ) -> pd.DataFrame:
        df = pd.DataFrame()
        for col in dx.columns:
            df[col] = f_inv(
                dx[col].values,
                centralized_records.centralized.values
            )
        df.index = dx.index
        return df

    return (stationarizer, inverse_stationarizer)


def to_stationary(
    centralizer: Callable,
    stationarizer: Callable,
    records: pd.DataFrame
) -> pd.DataFrame:
    """
    Generates a probably-stationary set of time series from a centralizing function,
    a stationarizing function, and a set of possibly-nonstationary time series.

    + `centralizer`: some function of central tendency. Must broadcast over arrays.
        Possible candidates are `np.mean` or `np.median`.
    + `stationarizer`: a function that attempts to generate wide-sense stationarity
        from a nonstationary time series. Probably generated as the first element of the
        tuple returned by a call to `make_stationarizer`.
    + `records`: `pd.DataFrame` with *evenly sampled* `datetime.datetime` index and `>=`
        non-negative numerical column(s). If this is not evenly sampled a result will
        still be returned, but may have adverse downstream effects.
    """
    centralized_records = pd.DataFrame()
    centralized_records["centralized"] = records.apply(centralizer, axis=1)
    return (
        stationarizer(records, centralized_records),
        centralized_records
    )