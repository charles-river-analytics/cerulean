import collections
import logging
from typing import Callable, Optional, Union

import mypy
import pyro
import pyro.distributions as dist
import pyro.distributions.constraints as constraints
from pyro.ops.contract import einsum
import torch


def joint_conditioned(eq: str, *tensors: torch.Tensor):
    return einsum(eq, *tensors, modulo_total=True)[0]


def marginal(eq: str, *tensors: torch.Tensor):
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


def make_factor_name(fs):
    return f"f_{fs}"


def factor_model(
    fs2dim: collections.OrderedDict[str,tuple[int,...]],
    data: Optional[collections.OrderedDict[str,torch.Tensor]]=None,
    query_var: Optional[str]=None
) -> Optional[torch.Tensor]:
    factors = collections.OrderedDict()
    for fs, dim in fs2dim.items():
        factors[fs] = pyro.param(
            make_factor_name(fs),
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

def query(
    model: Callable,
    fs2dim: collections.OrderedDict[str,tuple[int,...]],
    variables: str
) -> torch.Tensor:
    return model(fs2dim, query_var=variables)


class Factor:

    def __init__(
        self,
        name: str,
        fs: str,
        dim: Union[torch.Size, tuple[int,...]],
        table: Optional[torch.Tensor]=None,
    ):
        if len(fs) != len(dim):
            raise ValueError(
                    f"Dimension specification {fs} doesn't match dims {dim}"
                )
        self.name = name
        self.fs = fs
        self.dim = dim
        if (table is not None) and (self.dim != table.shape):
            raise ValueError(
            f"Dims {dim} don't match table shape {table.shape}"
        )
        self.table = table
        self.shape = None if self.table is None else self.table.shape

        self._variables = [var for var in self.name]
        self._variables_to_axis = collections.OrderedDict({
            var: i for (i, var) in enumerate(self._variables)
        })

    def __repr__(self,):
        s = "Factor("
        s += f"name={self.name}, "
        s += f"fs={self.fs}, "
        s += f"dim={self.dim})"
        return s

    def get_table(self,):
        return self.table

    def get_factor(self,):
        return pyro.param(make_factor_name(self.fs))
    
    def _post_evidence(self, var: int, level: int):
        if not self.table:
            raise ValueError(f"Factor {self.name}'s table has not been initialized!")
        if var in self._variables:
            axis = self._variables_to_axis[var]
            new_table = torch.index_select(
                self.table,
                axis,
                torch.tensor(level).type(torch.long)
            )
            new_table /= torch.sum(new_table)
            new_shape = new_table.shape

        return Factor(
            self.name,
            self.fs,
            new_shape,
            new_table,
        )


class FactorGraph:

    count = 0
    
    @classmethod
    def _next_id(cls):
        cls.count += 1
        return f"FactorGraph{cls.count}"

    @classmethod
    def learn(
        cls,
        fs2dim: collections.OrderedDict[str, tuple[int,...]],
        data: collections.OrderedDict[str, torch.Tensor],
    ):
        losses = mle_train(
            factor_model,
            (fs2dim,),
            dict(data=data),
        )
        factors = [
            Factor(f"Factor({fs})", fs, dim, pyro.param(make_factor_name(fs)))
            for (fs, dim) in fs2dim.items()
        ]
        new_factor_graph = cls(*factors)
        assert new_factor_graph.fs2dim == fs2dim
        return (new_factor_graph, losses)

    def __init__(self, *factors: Factor,):
        self.factors = collections.OrderedDict({
            factor.fs: factor for factor in factors
        })
        self.id = type(self)._next_id()
        self.fs2dim = collections.OrderedDict({
            factor.fs: factor.dim for factor in self.factors.values()
        })

        self._evidence_cache = list()

    def __repr__(self,):
        s = f"FactorGraph(id={self.id}\n"
        for f in self.factors.values():
            s += f"\t{f},\n"
        s +=")"
        return s

    def get_factor(self, fs: str):
        return self.factors[fs].get_factor()

    def post_evidence(self, var: str, level: int):
        ev = (var, level)
        if ev not in self._evidence_cache:
            for name in self.factors.keys():
                self.factors[name] = self.factors[name]._post_evidence(var, level)
            self._evidence_cache.append(ev)
        else:
            raise ValueError(f"Already posted evidence {ev}.")

    def query(self, variables: str):
        result_table = factor_model(self.fs2dim, query_var=variables)
        return Factor(
            f"{self.id}-query={variables}",
            variables,
            result_table.shape,
            result_table
        )