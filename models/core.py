import collections
import logging
from typing import Callable, Optional

import mypy
import pyro
import pyro.distributions as dist
import pyro.distributions.constraints as constraints
from pyro.ops.contract import einsum
import torch

def joint_conditioned(eq: str, *tensors):
    return einsum(eq, *tensors, modulo_total=True)[0]


def marginal(eq: str, *tensors):
    """
    Computes the exact marginal distribution via (cached) variable elimination.
    """
    unscaled = joint_conditioned(eq, *tensors)
    return (unscaled / torch.sum(unscaled))


def build_network_string(fs_list: list[str]):
    net_string = ""
    for fs in fs_list[:-1]:
        net_string += f"{fs},"
    net_string += fs_list[-1]
    return net_string


def factor_model(
    fs2dim: collections.OrderedDict[str,tuple[int,int]],
    data: Optional[collections.OrderedDict[str,torch.Tensor]]=None,
    query_var: Optional[str]=None
) -> Optional[torch.Tensor]:
    factors = collections.OrderedDict()
    for fs, dim in fs2dim.items():
        factors[fs] = pyro.param(
            f"f_{fs}",
            torch.ones(dim),
            constraint=constraints.positive
        )
    network_string = build_network_string(list(fs2dim.keys()))

    if not query_var:
        for var in fs2dim.keys():
            pr = marginal(f"{network_string}->{var}", *factors.values())
            with pyro.plate(f"{var}-plate") as ix:
                pyro.sample(
                    f"{var}-data",
                    dist.Categorical(pr.reshape((-1,))),
                    obs=data[var]
                )
    else:
        with torch.no_grad():
            return marginal(f"{network_string}->{query_var}", *factors.values())

def mle_train(
    model: Callable,
    model_args: tuple,
    model_kwargs: dict,
    num_iterations: int=1000,
    lr: float=0.01,
) -> torch.Tensor:
    guide = lambda *args, **kwargs: None
    opt = pyro.optim.Adam(dict(lr=lr))
    loss = pyro.infer.Trace_ELBO()
    svi = pyro.infer.SVI(model, guide, opt, loss=loss)
    
    pyro.clear_param_store()
    losses = torch.empty((num_iterations,))
    
    for j in range(num_iterations):
        loss = svi.step(*model_args, **model_kwargs)
        if j % 100 == 0:
            logging.info(f"On iteration {j}, -ELBO = {loss}")
        losses[j] = loss
    return losses
