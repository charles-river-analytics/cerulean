import collections
import datetime
import logging
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
        try:
            flattened_index_values = np.ravel_multi_index(
                factor_index_values,
                dimension_tuple
            )
        except ValueError:
            the_error = f"Can't unravel factors with dims {dimension_tuple},"
            the_error += " does the discretization of data in the generator"
            the_error += " align with these dimensions?"
            logging.error(the_error)
            raise ValueError(the_error)
        out["".join(variable_list)] = torch.Tensor(flattened_index_values)
    return out


class _Transform:
    """
    Bundles a transform with a name. 
    Not recommended for public use as the API could change at any time.
    """

    def __init__(self, name, transform,):
        self.name = name
        self._transform = transform

    def __call__(self, *args, **kwargs):
        return self._transform(*args, **kwargs)


_TRANSFORMS = {
    "vector": {
        "diff": {
            "f": lambda x, y: x[1:] - y[:-1],
            "f_inv": lambda dx, y: y[:-1] + dx
        },
        "logdiff": {
            "f": lambda x, y: 100.0 * (np.log(x[1:]) - np.log(y[:-1])),
            "f_inv": lambda dx, y: np.exp(dx / 100.0 + np.log(y[:-1]))
        }
    },
    "scalar": {
        "diff": {
            "f": lambda x, y: x - y,
            "f_inv": lambda dx, y: y + dx
        },
        "logdiff": {
            "f": lambda x, y: 100.0 * (np.log(x) - np.log(y)),
            "f_inv": lambda dx, y: np.exp(dx / 100.0 + np.log(y))
        }
    }
}


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
    f = _TRANSFORMS["vector"][method]["f"]
    f_inv = _TRANSFORMS["vector"][method]["f_inv"]

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

    return (_Transform(method, stationarizer), _Transform(method, inverse_stationarizer))


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
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Computes a discretization inferring min and max from data.
    """
    to_df = dict()
    for col in continuous_df.columns:
        # TODO: use the freqs for anything?
        the_freqs, the_bins = np.histogram(continuous_df[col], bins=n_bins)
        the_ixs = np.digitize(continuous_df[col].values, the_bins, right=True)
        to_df[col] = the_ixs
    return pd.DataFrame(to_df), the_bins


def _continuous_to_specific_variable_level(
    continuous_df: pd.DataFrame,
    n_bins: int,
    the_min: float,
    the_max: float
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Computes a discretization with a user-specified min and max.
    """
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
    return pd.DataFrame(to_df), the_bins


def continuous_to_variable_level(
    continuous_df: pd.DataFrame,
    n_cutpoints: int,
    the_min: Optional[float]=None,
    the_max: Optional[float]=None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Discretizes a collection of continuous rvs (represented via samples in 
    `pd.DataFrame`) into a collection of discrete rvs each of which has support 
    on :math:`\{0,1,2...,D - 1\}`, where :math:`D` `=n_cutpoints`. 
    
    If `the_min` and `the_max` aren't passed, min and max are inferred from 
    the data to be the min and max respectively of the sample. If you do not have 
    reason to believe that your sample contains something close to the actual min 
    and max possible of the DGP, you should probably pass an explicit min and max. 

    TODO: specify different number of cutpoints for each variable?
    """
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


def _stationary_to_nonstationary_bins(inverse_stationarizer, centralized, bins: np.ndarray) -> np.ndarray:
    """
    """
    the_inv = _Transform(
        inverse_stationarizer.name,
        _TRANSFORMS["scalar"][inverse_stationarizer.name]["f_inv"]
    )
    return the_inv(bins, centralized)
    

def get_names2strings(*names: str) -> dict[str, str]:
    """
    Maps variable names (which may be arbitrary strings) to unique unicode
    strings via `opt_einsum.parser`.
    """
    return {
        name: get_symbol(i) for (i, name) in enumerate(names)
    }
