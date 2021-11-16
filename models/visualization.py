
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import torch

from . import factor


DEFAULT_OUTPATH = pathlib.Path("figures")
DEFAULT_OUTPATH.mkdir(parents=True, exist_ok=True)


def _probability_compare_1d(
    variables: str,
    true: np.ndarray,
    pred: np.ndarray,
    outpath,
):
    assert len(true) == len(pred)
    fig, ax = plt.subplots()
    ax.bar(
        range(len(true)),
        true,
        color="white",
        alpha=0.3,
        label="True",
        width=0.5,
        hatch="//",
        edgecolor="black",
    )
    ax.bar(
        range(len(pred)),
        pred,
        color="white",
        alpha=0.3,
        label="Predicted",
        width=0.5,
        hatch="\\",
        edgecolor="black",
    )
    ax.set_xlabel(f"Level of {variables}")
    ax.set_ylabel(f"p({variables})")
    ax.legend()
    plt.savefig(outpath / f"{variables}-marginal.png")


def _probability_compare_2d(
    variables: str,
    true: np.ndarray,
    pred: np.ndarray,
    outpath,
):
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(true, interpolation="none")
    axes[1].imshow(pred, interpolation="none")

    plt.savefig(outpath / f"{variables}-marginal.png")


def probability_compare(
    fg: factor.DiscreteFactorGraph,
    variables: str,
    true: np.ndarray,
    outpath=DEFAULT_OUTPATH,
):
    len_variables = len(variables)
    if len_variables > 2:
        raise ValueError(
            "Only plotting univariate and bivariate dists are supported."
        )

    prob_factor = fg.query(variables)
    preds = prob_factor.table.numpy()

    if len_variables == 1:
        _probability_compare_1d(variables, true, preds, outpath)
    else:
        _probability_compare_2d(variables, true, preds, outpath)
