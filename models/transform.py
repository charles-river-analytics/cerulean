import collections
import datetime
from typing import Callable, Iterable, Literal, Optional

import mypy
import numpy as np
from opt_einsum.parser import get_symbol
import pandas as pd
import torch


StationaryTransform = Literal["diff", "logdiff"]


def _df2od_torch(
    df: pd.DataFrame,
    variables: Iterable[Iterable[str]],
    dimensions: Iterable[tuple[int,...]]
) -> collections.OrderedDict[str, torch.Tensor]:
    """
    Convert dataframe data with observations by variable to flattened observations
    suitable for indexing into a factor (used for evaluating log probability).

    + `df`: dataframe of data. Column names should be (string) variables, values in
        columns should be (non-negative) integer levels of the variable. The dataframe
        should have equal length columns and there should be no NaN values. 
    + `variables`: each outer list corresponds to a unique factor, while each 
        inner list is a list of string variables contained in the factor. 
    + `dimensions`:  each tuple is that factor's dimensions. The length of the tuple
        is equal to the degree of the factor.
    """
    out = collections.OrderedDict()
    # each (variable, dimension) corresponds to a single factor's info
    for (variable_list, dimension_tuple) in zip(variables, dimensions):
        factor_index_values = [df[k].values for k in variable_list]
        flattened_index_values = np.ravel_multi_index(
            factor_index_values,
            dimension_tuple
        )
        out["".join(variable_list)] = torch.Tensor(flattened_index_values)
    return out


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


def _continuous_to_auto_variable_level(
    continuous_df: pd.DataFrame,
    n_bins: int
) -> pd.DataFrame:
    to_df = dict()
    for col in continuous_df.columns:
        # TODO: use the freqs for anything?
        the_freqs, the_bins = np.histogram(continuous_df[col], bins=n_bins)
        the_ixs = np.digitize(continuous_df[col].values, the_bins, right=True)
        to_df[col] = the_ixs
    return pd.DataFrame(to_df)


def _continuous_to_specific_variable_level(
    continuous_df: pd.DataFrame,
    n_bins: int,
    the_min: float,
    the_max: float
) -> pd.DataFrame:
    to_df = dict()
    for col in continuous_df.columns:
        # TODO: use the freqs for anything?
        the_freqs, the_bins = np.histogram(
            continuous_df[col],
            bins=n_bins,
            range=(the_min, the_max)
        )
        the_ixs = np.digitize(continuous_df[col].values, the_bins, right=True)
        to_df[col] = the_ixs
    return pd.DataFrame(to_df)


def continuous_to_variable_level(
    continuous_df: pd.DataFrame,
    n_cutpoints: int,
    the_min: Optional[float]=None,
    the_max: Optional[float]=None,
) -> pd.DataFrame:
    if (the_min is None) & (the_max is not None) | (the_min is not None) & (the_max is None):
        raise ValueError("Both the_min and the_max must be None, or neither.")
    n_bins = max(n_cutpoints - 1, 2)
    if the_max is None:
        return _continuous_to_auto_variable_level(continuous_df, n_bins)
    else:
        return _continuous_to_specific_variable_level(
            continuous_df,
            n_bins,
            the_min,
            the_max
        )


def get_names2strings(*names: str) -> dict[str, str]:
    return {
        name: get_symbol(i) for (i, name) in enumerate(names)
    }
