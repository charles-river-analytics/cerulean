
import itertools
import logging
import pathlib
import time
from typing import Iterable

import cerulean
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import torch

from stationary import corr_asset_model


logging.basicConfig(level=logging.INFO)
DEFAULT_OUTPATH = pathlib.Path("random_inference")
DEFAULT_OUTPATH.mkdir(parents=True, exist_ok=True)


def make_random_chain(
    chain_length: int,
    dim: int,
):
    assert chain_length >= 2
    factory = cerulean.dimensions.DimensionsFactory(
        *(f"var_{t}" for t in range(chain_length))
    )
    for t in range(chain_length):
        factory(f"var_{t}", dim)
    the_factor_dims = [
        factory((f"var_{t-1}", f"var_{t}"))
        for t in range(1, chain_length)
    ]
    the_factors = [
        cerulean.constraint.ConstraintFactor.random(fd)
        for fd in the_factor_dims
    ]
    graph = cerulean.factor.DiscreteFactorGraph(*the_factors)
    return (the_factor_dims, graph)


def time_chain_inference(
    num_iterations: int,
    dims: Iterable[cerulean.dimensions.FactorDimensions],
    graph: cerulean.factor.DiscreteFactorGraph,
    num_select: int
):
    times = []
    for n in range(num_iterations):
        the_random_dims = np.random.choice(dims, size=num_select)
        for dim in the_random_dims:
            v_str = dim.get_variable_str()
            t0 = time.time()
            _ = graph.query(v_str)
            t1 = time.time()
            micro_time = round(1e6 * (t1 - t0), 3)
            times.append(micro_time)
    return (np.mean(times), np.std(times))


def chain_inference_scaling():
    chain_lengths = [2, 10, 25]# 50, 100]
    reruns = [50, 10, 4] #2, 1]
    # chain_length * rerun = 100
    dims = [3, 5, 10, 50, 100]# 250]
    df = pd.DataFrame(
        columns=chain_lengths,
        index=dims,
    )
    for (ix, cl) in enumerate(chain_lengths):
        for dim in dims:
            logging.info(f"Chain length = {cl}, dim = {dim}")
            d, g = make_random_chain(cl, dim)
            mean_t, std_t = time_chain_inference(reruns[ix], d, g, 20)
            df[cl].loc[dim] = mean_t

    df = df.apply(pd.to_numeric)
    return df


def plot_chain_scaling(results_df):
    fig, ax = plt.subplots()
    im = ax.imshow(
        results_df.values,
        interpolation="none",
        cmap="cool",
        aspect="auto",
        norm=matplotlib.colors.LogNorm(),
    )
    ax.set_xticks(range(len(results_df.columns)))
    ax.set_yticks(range(len(results_df.index)))
    ax.set_xticks(np.arange(-.5, len(results_df.columns), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(results_df.index), 1), minor=True)
    ax.grid(
        which="minor",
        color="k",
        linestyle="-",
        linewidth=2,
    )
    ax.set_xticklabels(results_df.columns)
    ax.set_yticklabels(results_df.index)
    ax.set_xlabel("Chain length")
    ax.set_ylabel("Variable dimension")

    cbar = fig.colorbar(
        im,
        ax=ax,
    )
    cbar.ax.set_yscale("log")
    cbar.ax.set_ylabel("Clock time (us)",)

    for (j,i), label in np.ndenumerate(results_df.values):
        ax.text(i, j, round(label), ha='center', va='center')

    plt.savefig(DEFAULT_OUTPATH / "chain_scaling.png")
    plt.savefig(DEFAULT_OUTPATH / "chain_scaling.pdf")
    plt.close()


def make_asset_price_ts(n_ts):
    num_assets = 4
    ic = 100.0
    paths = corr_asset_model(num_assets, n_ts, torch.tensor(ic))
    paths_df = pd.DataFrame(
        paths.T.numpy()
    )
    return paths_df


def make_evidence_ts(n_ts:int, n_cutpoints: int):
    asset_price = make_asset_price_ts(n_ts)
    forward_s, partial_inverse_s = cerulean.transform.make_stationarizer(
        "logdiff"
    )
    # centralize with mean
    stationarized_paths_df, centralized_df = cerulean.transform.to_stationary(
        np.mean,
        forward_s,
        asset_price,
    )
    # note: we would create this from variabledimensions usually
    stationarized_paths_df.columns = ["a", "b", "c", "d"]
    the_min = stationarized_paths_df.values.min() * 1.1
    the_max = stationarized_paths_df.values.max() * 1.1
    evidence = cerulean.transform.continuous_to_variable_level(
        stationarized_paths_df,
        n_cutpoints=n_cutpoints,
        the_min=the_min,
        the_max=the_max
    )
    return evidence


def setup_market_model(n_cutpoints: int):
    n_ts = 100
    names = ["A", "B", "C", "D"]
    factory = cerulean.dimensions.DimensionsFactory(*names)
    for name in names:
        factory(name, n_cutpoints=n_cutpoints)
    factor_dims = [factory(d) for d in itertools.combinations(names, 2)]

    data = make_evidence_ts(n_ts, n_cutpoints)

    # smoketest -- just "train" to get model
    # we need to speed up training by pre-computing path *once* during
    # training, then using that path for all of training...
    trained_model = cerulean.factor.DiscreteFactorGraph.learn(
        factor_dims,
        data,
        train_options=dict(verbosity=20, num_iterations=100)
    )
    return trained_model


def time_market_model_infer_1(
    market_model: cerulean.factor.DiscreteFactorGraph,
    n_cutpoints: int
):
    # do inference for one variable after observing all others
    new_model = market_model.snapshot()
    new_model.post_evidence("a", int(n_cutpoints/2))
    new_model.post_evidence("b", int(n_cutpoints/2))
    new_model.post_evidence("c", int(n_cutpoints/2))
    t0 = time.time()
    _ = new_model.query("d")
    t1 = time.time()
    return round(1e6 * (t1 - t0))


def time_market_model_infer_nminus1(
    market_model: cerulean.factor.DiscreteFactorGraph,
    n_cutpoints: int
):
    new_model = market_model.snapshot()
    new_model.post_evidence("d", int(n_cutpoints/2))
    t0 = time.time()
    _ = new_model.query("abc")
    t1 = time.time()
    return round(1e6 * (t1 - t0))


def market_model_scaling_data(num_reruns: int):
    ctpts = [5, 9, 15, 31, 61, 101,] 
    columns = ["Infer 1", "Infer N-1"]
    fns = [
        time_market_model_infer_1,
        time_market_model_infer_nminus1
    ]
    df = pd.DataFrame(columns=columns, index=ctpts)
    for n_cutpoints in ctpts:
        logging.info(f"Doing market inference with {n_cutpoints} discretization")
        trained_model, loss_values = setup_market_model(n_cutpoints)
        cerulean.visualization.plot_losses(loss_values, outpath=DEFAULT_OUTPATH,)
        for (col, fn) in zip(columns, fns):
            vals = []
            for n in range(num_reruns):
                tval = fn(trained_model, n_cutpoints)
                vals.append(tval)
            df[col].loc[n_cutpoints] = np.mean(vals)
            logging.info(f"With fn {col}, average time is {df[col].loc[n_cutpoints]}")
    df = df.apply(pd.to_numeric)
    return df


def plot_market_model_scaling(results_df):
    fig, ax = plt.subplots()
    im = ax.imshow(
        results_df.values,
        interpolation="none",
        cmap="cool",
        aspect="auto",
        norm=matplotlib.colors.LogNorm(),
    )
    ax.set_xticks(range(len(results_df.columns)))
    ax.set_yticks(range(len(results_df.index)))
    ax.set_xticks(np.arange(-.5, len(results_df.columns), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(results_df.index), 1), minor=True)
    ax.grid(
        which="minor",
        color="k",
        linestyle="-",
        linewidth=2,
    )
    ax.set_xticklabels(results_df.columns)
    ax.set_yticklabels(results_df.index)
    ax.set_xlabel("Query")
    ax.set_ylabel("Discretization size")

    cbar = fig.colorbar(
        im,
        ax=ax,
    )
    cbar.ax.set_yscale("log")
    cbar.ax.set_ylabel("Clock time (us)",)

    for (j,i), label in np.ndenumerate(results_df.values):
        ax.text(i, j, round(label), ha='center', va='center')

    plt.savefig(DEFAULT_OUTPATH / "market_scaling.png")
    plt.savefig(DEFAULT_OUTPATH / "market_scaling.pdf")
    plt.close()


def main():
    logging.info("Chain inference scaling...")
    chain_results = chain_inference_scaling()
    logging.info(chain_results)
    plot_chain_scaling(chain_results)

    logging.info("Market inference scaling...")
    mm_results = market_model_scaling_data(20)
    logging.info(mm_results)
    plot_market_model_scaling(mm_results)


if __name__ == "__main__":
    main()