
import logging
import pathlib
import sys

logging.basicConfig(level=logging.INFO)

import numpy as np
import matplotlib.pyplot as plt
import opt_einsum
import pandas as pd
import pyro
import pyro.distributions as dist
from scipy import signal, stats
import torch

import cerulean


SAVEDIR = pathlib.Path("./stationary")
SAVEDIR.mkdir(exist_ok=True, parents=True,)


def corr_asset_model(
    num_assets: int,
    num_ts: int,
    ic: torch.Tensor,
) -> torch.Tensor:
    corr_mat_base = pyro.sample(
        "corr_mat_base",
        dist.Uniform(0.99, 0.999).expand((num_assets, num_assets))
    )
    corr_mat_base = torch.mm(corr_mat_base, corr_mat_base.t()) / num_assets
    corr_mat_base.fill_diagonal_(1.0)
    corr_mat = pyro.deterministic(
        "corr_mat",
        corr_mat_base
    )
    logging.info(f"Correlation matrix = {corr_mat}")
    # marginal volatility
    log_vol_vector = pyro.sample(
        "log_vol", 
        dist.Normal(
            torch.log(torch.tensor(0.01)),
            0.25
        ).expand((num_assets,))
    )
    vol = log_vol_vector.exp()

    cov = opt_einsum.contract(
        "i,ij,j->ij",
        vol, corr_mat, vol
    )
    logging.info(f"Covariance matrix = {cov}")
    correlated_innovations = pyro.sample(
        "correlated_innovations",
        dist.MultivariateNormal(
            torch.zeros(num_assets),
            covariance_matrix=cov
        ).expand((num_ts,))
    ).T
    log_ic = torch.log(ic)
    asset_price = pyro.deterministic(
        "asset_price",
        (correlated_innovations.cumsum(dim=-1) + log_ic).exp()
    )
    return asset_price


def linear_regression_1d(X, y):
    slope = pyro.sample(
        "beta",
        dist.Normal(0.0, 1.0)
    )
    intercept = pyro.sample(
        "alpha",
        dist.Normal(0.0, 3.0)
    )
    sigma = pyro.sample(
        "sigma",
        dist.LogNormal(0.0, 1.0)
    )
    mu = X * slope + intercept
    with pyro.plate("data_plate") as plate:
        pyro.sample(
            "data",
            dist.Normal(mu, sigma),
            obs=y,
        )


def run_ols1d_inference(X, y, num_training=2500,):
    guide = pyro.infer.autoguide.AutoNormal(linear_regression_1d)
    optim = pyro.optim.Adam({"lr": 0.05})
    loss = pyro.infer.Trace_ELBO()
    svi = pyro.infer.SVI(linear_regression_1d, guide, optim, loss=loss)

    pyro.clear_param_store()

    for n in range(num_training):
        loss = svi.step(X, y,)
        if n % 250 == 0:
            logging.info(f"OLS 1d train step {n}\n-ELBO = {loss}\n")

    return pyro.infer.Predictive(
        linear_regression_1d,
        guide=guide,
        num_samples=1000,
    )


def compute_spectrogram_exponents(paths):
    means = []
    stds = []
    for path in paths:
        f, t, s = signal.spectrogram(path)
        log_f = torch.tensor(np.log(f[1:].squeeze()))
        log_s = torch.tensor(np.log(s[1:].squeeze()))
        pred_model = run_ols1d_inference(
            log_f,
            log_s
        )
        preds = pred_model(
            log_f,
            log_s
        )
        means.append(-1.0 * preds["beta"].mean())
        stds.append(preds["beta"].std())
    return torch.tensor(means), torch.tensor(stds)


def plot_time_series(paths, st_paths):
    fig, axes = plt.subplots(1, 2)
    for ax in axes:
        ax.grid("on")
    axes[0].plot(
        paths,
        color="dodgerblue",
    )
    axes[0].set_title("Original")
    axes[1].plot(
        st_paths,
        color="black",
    )
    axes[1].set_title("Stationarized")

    for ax in axes:
        ax.set_xlabel("Operational time")

    axes[0].set_ylabel("Asset price")
    axes[1].set_ylabel("Transformed")
    fig.tight_layout()

    plt.savefig(SAVEDIR / "ts_paths.png")
    plt.close()


def plot_psd_exps(means, stds, st_means, st_stds):
    fig, ax = plt.subplots()
    ax.grid("on")
    for (m, s) in zip(means, stds):
        the_values = np.linspace(m - 5 * s, m + 5 * s, 250)
        the_pdf = stats.norm(m, s).pdf(the_values)
        ax.plot(
            the_values,
            the_pdf,
            color="dodgerblue",
        )
    ax.plot(
        the_values,
        the_pdf,
        color="dodgerblue",
        label="Original"
    )
    for (m, s) in zip(st_means, st_stds):
        the_values = np.linspace(m - 5 * s, m + 5 * s, 250)
        the_pdf = stats.norm(m, s).pdf(the_values)
        ax.plot(
            the_values,
            the_pdf,
            color="black",
        )
    ax.plot(
        the_values,
        the_pdf,
        color="black",
        label="Stationarized",
    )
    ax.axvline(
        1.0,
        color="red",
    )

    ax.set_xlabel("PSD exponent (b)")
    ax.set_ylabel("p(b)")
    ax.legend()

    plt.savefig(SAVEDIR / "psd_exps.png")
    plt.close()
        

def main():
    num_assets = 5
    num_ts = 250
    ic = torch.tensor(100.0)

    # GBM asset prices
    paths = corr_asset_model(num_assets, num_ts, ic)
    paths_df = pd.DataFrame(
        paths.T.numpy()
    )
    logging.info(f"Original paths:\n{paths_df}")
    logging.info("Now learning PSD exponent distributions for original...")
    means, stds = compute_spectrogram_exponents(paths)

    # stationarizer
    forward_s, partial_inverse_s = cerulean.transform.make_stationarizer(
        "logdiff"
    )
    # centralize with mean
    stationarized_paths_df, centralized_df = cerulean.transform.to_stationary(
        np.mean,
        forward_s,
        paths_df,
    )
    logging.info(f"Stationarized paths:\n{stationarized_paths_df}")
    logging.info("Now learning PSD exponent distributions for stationarized...")
    st_means, st_stds = compute_spectrogram_exponents(stationarized_paths_df.values.T)

    plot_time_series(paths_df, stationarized_paths_df)
    plot_psd_exps(means, stds, st_means, st_stds)


if __name__ == "__main__":
    main()
