
import pathlib
from typing import Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from . import factor


DEFAULT_OUTPATH = pathlib.Path("figures")
DEFAULT_OUTPATH.mkdir(parents=True, exist_ok=True)
SINGLE_FIGSIZE = (8, 5)
DOUBLE_FIGSIZE = (5, 5)
FONTSIZE = 15


def _probability_compare_1d(
    variables: str,
    true: np.ndarray,
    pred: np.ndarray,
    outpath,
    labels,
    log,
):
    assert len(true) == len(pred)
    labels = np.arange(len(true))
    width = 0.35
    fig, ax = plt.subplots(figsize=SINGLE_FIGSIZE)
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
    ax.set_xticks(labels,)
    ax.set_xlabel(f"Level of {variables}", fontsize=FONTSIZE)
    ax.set_ylabel(f"p({variables})", fontsize=FONTSIZE)
    ax.legend()
    plt.savefig(outpath / f"{variables}-marginal.png")
    plt.close()


def _probability_compare_2d(
    variables: str,
    true: np.ndarray,
    pred: np.ndarray,
    outpath,
    labels,
    log,
):
    var_1, var_2 = tuple(x for x in variables)
    fig, axes = plt.subplots(2, 1, )#figsize=SINGLE_FIGSIZE)
    axes = axes.flatten()
    if not log:
        axes[0].imshow(
            true,
            interpolation="none",
            cmap="autumn",
            aspect="auto",
        )
        true_display = axes[1].imshow(
            pred,
            interpolation="none",
            cmap="autumn",
            aspect="auto",
        )
    else:
        norm = matplotlib.colors.LogNorm()
        axes[0].imshow(
            true,
            interpolation="none",
            cmap="autumn",
            aspect="auto",
            norm=norm,
        )
        true_display = axes[1].imshow(
            pred,
            interpolation="none",
            cmap="autumn",
            aspect="auto",
            norm=norm,
        )
    axes[0].set_xticks([])
    axes[0].set_yticks(range(true.shape[0]))
    axes[1].set_yticks(range(true.shape[0]))
    axes[1].set_xticks(range(true.shape[1]))

    axes[0].set_ylabel(f"Level of {var_1}", )#fontsize=FONTSIZE)
    axes[1].set_ylabel(f"Level of {var_1}", )#fontsize=FONTSIZE)
    axes[1].set_xlabel(f"Level of {var_2}", )#fontsize=FONTSIZE)

    cbar = fig.colorbar(
        true_display,
        ax=axes.ravel().tolist()
    )
    cbar.ax.set_ylabel(
        f"p({','.join([x for x in variables])})",
        fontsize=FONTSIZE
    )

    axes[0].set_title("Empirical")
    axes[1].set_title("Predicted")

    if labels:
        for ax in axes:
            ax.set_xticks(np.arange(-.5, true.shape[1], 1), minor=True)
            ax.set_yticks(np.arange(-.5, true.shape[0], 1), minor=True)
            ax.grid(
                which="minor",
                color="k",
                linestyle="-",
                linewidth=2,
            )
        for (j,i), label in np.ndenumerate(true):
            axes[0].text(i, j, round(label, 3), ha='center', va='center')
        for (j,i), label in np.ndenumerate(pred):
            axes[1].text(i, j, round(label, 3), ha='center', va='center')
    if outpath is False:
        return (fig, axes)
    plt.savefig(outpath / f"{variables}-marginal.png")
    plt.close()


def plot_losses(
    losses: torch.Tensor,
    outpath=DEFAULT_OUTPATH,
):
    """
    Plot losses from training a `FactorGraph`.
    """
    fig, ax = plt.subplots()
    ax.grid("on")
    ax.plot(
        losses.numpy(),
        color="royalblue",
    )
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Loss value")
    fig.tight_layout()
    plt.savefig(outpath / f"losses.png")
    plt.close()


def probability_compare(
    fg: factor.DiscreteFactorGraph,
    variables: str,
    true: np.ndarray,
    outpath: Union[str, bool]=DEFAULT_OUTPATH,
    labels: bool=True,
    log: bool=False,
):
    """
    Plot univariate and bivariate probability distributions and compare them 
    with data. 

    + `fg`: a factor graph. The `variables` will be passed to `.query(...)`,
        computing a marginal (possibly conditional) probability distribution.
    + `variables`: the marginal query to run.
    + `true`: an array of numbers representing the "true" or empirical probability distribution.
    + `outpath`: location to save figures
    """
    len_variables = len(variables)
    if len_variables > 2:
        raise ValueError(
            "Only plotting univariate and bivariate dists are supported."
        )

    prob_factor = fg.query(variables)
    preds = prob_factor.table.numpy()

    if len_variables == 1:
        return _probability_compare_1d(variables, true, preds, outpath, labels, log,)
    else:
        return _probability_compare_2d(variables, true, preds, outpath, labels, log,)
