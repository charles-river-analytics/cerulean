
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
    labels = np.arange(len(true))
    width = 0.35
    fig, ax = plt.subplots()
    pred_bars = ax.bar(
        labels - width/2, pred, width,
        label="Predicted",
        hatch="//",
        edgecolor="black"
    )
    ax.bar_label(
        pred_bars,
        padding=3,
        fmt="%.3f",
    )
    true_bars = ax.bar(
        labels + width/2, true, width,
        label="Empirical",
        hatch="\\",
        edgecolor="black"
    )
    ax.bar_label(true_bars, padding=3)
    ax.set_xticks(labels)
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
    fig, axes = plt.subplots(2, 1)
    axes = axes.flatten()
    axes[0].imshow(
        true,
        interpolation="none",
        cmap="cividis",
    )
    true_display = axes[1].imshow(
        pred,
        interpolation="none",
        cmap="cividis",
    )
    axes[0].set_xticks(range(true.shape[1]))
    axes[0].set_yticks(range(true.shape[0]))
    axes[1].set_xticks(range(true.shape[1]))
    axes[1].set_yticks(range(true.shape[0]))

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
    fig.colorbar(true_display, cax=cbar_ax)

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
