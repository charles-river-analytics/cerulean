
import abc
import collections
import datetime
import functools
import logging
from typing import (
    Callable,
    Iterable,
    Literal,
    Mapping,
    Optional,
    Union
)

import mypy
import numpy as np
import opt_einsum
import pandas as pd
import pyro
import pyro.distributions as dist
import pyro.distributions.constraints as constraints
import torch

from . import (
    dimensions,
    transform
)


def discrete_marginal(
    contract_op,
    *tensors: torch.Tensor
):
    """
    Computes the exact discrete_marginal distribution via (cached) variable elimination.

    This method computes 
    :math:`p(V, E = e)`, then computes
    :math:`p(V | E = e) = p(V, E = e) / p(E = e)`. Note that the complexity of this method
    scales as :math:`\mathcal{O}(D^{|V|})` where :math:`D` is the maximum variable
    support cardinality.
    """
    unscaled = contract_op(*tensors)
    return (unscaled / torch.sum(unscaled))


def make_factor_name(fs: str, prefix: str="f"):
    """
    Makes a factor name from a dimension specification string. 
    """
    return f"{prefix}_{fs}"


@functools.lru_cache(maxsize=128,)
def make_contract_expr(
    network_string: str,
    var: str,
    shapes,
    optimize="greedy",
):
    """
    Generates a contraction path for inferring the marginal distribution of
    `var` from the factor graph denoted by `network_str` that consists of factors
    with shapes given by `shapes`. 

    NOTE: this method uses an LRU cache to store contraction paths. Paths are mapped to by
    a (network_string, var, shapes) tuple.
    """
    return opt_einsum.contract_expression(
        network_string + f"->{var}",
        *shapes,
        optimize=optimize,
    )


def contraction_cache_info():
    """
    Returns the (hits, misses, maxsize, currsize) of the contraction path cache.
    """
    return make_contract_expr.cache_info()


def contraction_cache_status(f):
    """
    Wraps the callable `f`, reporting cache status before and after `f` is executed.
    """
    def wrapper(*args, **kwargs):
        logging.info(
            f"Cache status before {f}: {contraction_cache_info()}"
        )
        rval = f(*args, **kwargs)
        logging.info(
            f"Cache status after {f}: {contraction_cache_info()}"
        )
        return rval
    return wrapper


def discrete_factor_model(
    fs2dim: collections.OrderedDict[str,tuple[int,...]],
    data: Optional[collections.OrderedDict[str,torch.Tensor]]=None,
    query_var: Optional[str]=None,
    factor_name_prefix: str="f",
    contract_expr=None,
) -> Optional[torch.Tensor]:
    """A Pyro model corresponding to a discrete factor graph. This model supports both MLE parameter
    learning and inference.

    This function takes a `collections.OrderedDict` of {dimension name string: dimension size tuple}, e.g., 
    {"ab": (2, 3), "bc": (3, 4)}.  

    If `data` is not None, then this function scores the observed data against the current values of the factors
    using Pyro machiner.

    If `contract_expr` is not None, it must be an `opt_einsum.ContractExpression` defining the algorithm by which 
    tensors are contracted. It is *not* checked that this is a valid contraction algorithm at runtime, nor is the
    algorithm guaranteed to be fast if it is valid. If in doubt, pass `contract_expr=None` which will compute a valid
    contraction at runtime.
    
    TODO: it should be possible to instead request a fully Bayesian treatment.
    Alternatively, if `query_var` is not None, this function performs exact inference using 
    (cached) variable elimination to find the marginal distribution of the query variables. For example, 
    in the factor graph implicitly defined by the `fs2dim` of `{"ab": (2, 3), "bc": (3, 4)}`, 
    setting `query_var="ac" would infer the marginal probability `p(a,c)`, while `query_var="b"` would infer the 
    marginal probability `p(b)` and `query_var="abc"` would compute the entire joint density `p(a,b,c)`.
    """
    factors = collections.OrderedDict()
    for fs, dim in fs2dim.items():
        factors[fs] = pyro.param(
            make_factor_name(fs, prefix=factor_name_prefix,),
            torch.ones(dim),
            constraint=constraints.positive
        )
    network_string = ",".join(fs2dim.keys())

    if not query_var:
        with opt_einsum.shared_intermediates():  # NOTE: automated only during learning!
            if contract_expr is None:
                for var in fs2dim.keys():  # iterate through all defined cliques
                    contract_expr = make_contract_expr(
                        network_string,
                        var,
                        tuple(map(lambda f: f.shape, factors.values()))
                    )
                    pr = discrete_marginal(contract_expr, *factors.values())
                    with pyro.plate(f"{var}-plate") as ix:
                        pyro.sample(
                            f"{var}-data",
                            dist.Categorical(pr.reshape((-1,))),
                            obs=data[var]
                        )
            else:
                for var in fs2dim.keys():  # iterate through all defined cliques
                    pr = discrete_marginal(contract_expr, *factors.values())
                    with pyro.plate(f"{var}-plate") as ix:
                        pyro.sample(
                            f"{var}-data",
                            dist.Categorical(pr.reshape((-1,))),
                            obs=data[var]
                        )
    else:
        with torch.no_grad():
            if contract_expr is None:
                contract_expr = make_contract_expr(
                    network_string,
                    query_var,
                    tuple(map(lambda f: f.shape, factors.values()))
                )
                return discrete_marginal(contract_expr, *factors.values())
            else:
                return discrete_marginal(contract_expr, *factors.values())


def _setup_svi(
    model,
    train_options: dict,
):
    opt_str = train_options.get("opt", "Adam")
    opt_cls = getattr(pyro.optim, opt_str)
    guide = lambda *args, **kwargs: None

    lr = train_options.get("lr", 0.01)
    opt = opt_cls(dict(lr=0.01))
    loss = pyro.infer.Trace_ELBO()
    svi = pyro.infer.SVI(model, guide, opt, loss=loss)
    return svi


def query(
    model: Callable,
    fs2dim: collections.OrderedDict[str,tuple[int,...]],
    variables: str
) -> torch.Tensor:
    """Runs the probabilistic query against the callable Pyro stochastic function. 
    See documentation of `discrete_factor_model` for more details. 
    """
    return model(fs2dim, query_var=variables)


class FactorNode(abc.ABC):
    """
    Abstract base class from which all factor nodes must inherit. 
    Factor nodes must implement a `snapshot` method, which should return
    a deep copy of the factor node without any linkage to global state 
    of any kind (e.g., param store, gradient tape), and a
    `_post_evidence` method which allows evidence to be posted to the
    factor and returns an instance of `cls`.
    """

    @abc.abstractmethod
    def snapshot(self,):
        ...

    @abc.abstractmethod
    def _post_evidence(self, var, level):
        ...


class DiscreteFactor(FactorNode):
    """A `DiscreteFactor` is an object-oriented wrapper around a `torch.tensor` that is
    interpreted as a factor node in a factor graph. 
    """

    def __init__(
        self,
        name: str,
        fs: str,
        dim: Union[torch.Size, tuple[int,...]],
        table: Optional[torch.Tensor]=None,
        factor_name_prefix: str="f",
    ):
        """
        A `DiscreteFactor` takes parameters:

        + `name`: name of the factor
        + `fs`: dimension labeling/specification. For example, if the factor relates variables `a` and `b`,
            we would have `fs = "ab"`. 
        + `dim`: the shape of the dimensions. For example, if the variable `a` could take on three values
            and the variable `b` could take on seventy, we would have `dim = (3, 70)`.
        + `table`: optional, corresponds to values of the factor (i.e., the values in the factor will not be
            learned from data). Defaults to `None` as usually this `__init__` method will be called from 
            the `DiscreteFactorGraph` class's `.learn(...)` method.
        """
        if len(fs) != len(dim):
            raise ValueError(
                    f"Dimension specification {fs} doesn't match dims {dim}"
                )
        self.fs = fs
        self.dim = dim
        if (table is not None) and (self.dim != table.shape):
            raise ValueError(
            f"Dims {dim} don't match table shape {table.shape}"
        )
        self.table = table
        self.shape = None if self.table is None else self.table.shape

        self._factor_name_prefix = factor_name_prefix
        self._variables = [var for var in self.fs]
        self._variables_to_axis = collections.OrderedDict({
            var: i for (i, var) in enumerate(self._variables)
        })
        self.name = self._factor_name_prefix + "_" + name

    def snapshot(self,):
        """
        Snapshots the state of a factor. Returns 
        a new factor with values identical to the original iff the original
        factor has a non-null table. Otherwise raises `ValueError` as this 
        method should be called only when a factor has been initialized.
        """
        if self.table is None:
            raise ValueError(
                "Can't take a snapshot of factor with uninitialized table"
            )
        return DiscreteFactor(
            self.name,
            self.fs,
            self.dim,
            self.table.clone().detach(),
        )

    def __repr__(self,):
        s = "DiscreteFactor("
        s += f"name={self.name}, "
        s += f"fs={self.fs}, "
        s += f"dim={self.dim})"
        return s

    def get_table(self,):
        """Returns the `Optional[torch.Tensor]` corresponding to the actual 
        factor (which is `None` if the factor itself hasn't been initialized)
        """
        return self.table

    def get_factor(self,):
        """Returns the factor as stored in Pyro's `param_store`.
        """
        return pyro.param(make_factor_name(self.fs, prefix=self._factor_name_prefix))
    
    def _post_evidence(self, var: str, level: int):
        """Posts evidence to the factor.

        Returns a new `DiscreteFactor` for which the table is equal to the view 
        of the Pyro parameter (`torch.Tensor`) corresponding to the observed value.
        For example, if a factor had `fs = "abc"` with corresponding dimensions
        `(2, 3, 4)` and this method were called with `var = "b"` and `level = 2`,
        a `DiscreteFactor` would be returned that had a `torch.Tensor` table with 
        shape equal to `(2, 1, 4)` with the singleton dimension corresponding to the 
        last-most slice of that dimension in the original tensor. 

        TODO: fix this docstring to be less confusing.
        NOTE: see documentation of `torch.index_select` which this method uses 
        internally.
        """
        if self.table is None:
            raise ValueError(f"DiscreteFactor {self.name}'s table has not been initialized!")
        if var in self._variables:
            axis = self._variables_to_axis[var]
            new_table = torch.index_select(
                self.table,
                axis,
                torch.tensor(level).type(torch.long)
            )
            new_shape = new_table.shape
            # note that the new table is *not* normalized
            # because this is now a likelihood function
            return DiscreteFactor(
                self.name,
                self.fs,
                new_shape,
                new_table,
            )
        else:  # TODO: should this be an error or a no-op (as it is now)?
            logging.debug(f"Variable {var} not in DiscreteFactor {self.name}")
            return self

    def kl_divergence(self, other: Union["DiscreteFactor", np.ndarray]) -> float:
        """
        Compute the Kullback-Leibler (KL) divergence between this factor and another.

        The KL divergence between this factor :math:`\psi` and another (normalized)
        factor :math:`\phi` is equal to

        .. math:: \mathrm{D}(\psi||\phi) = \sum_x \\frac{\psi(x)}{Z_\psi} 
            \left[ \log \\frac{\psi(x)}{Z_\psi} - \log \\frac{\phi(x)}{Z_\phi} \\right].

        The numbers :math:`Z` are the partition functions that convert the factors into proper
        probability distributions. Note that the complexity of this method is exponential in the 
        degree of the factor(s). NOTE: The logarithm is taken in base 2; the result has units of bits.
        """
        if type(other) == np.ndarray:
            if tuple(other.shape) != tuple(self.table.shape):
                raise ValueError(
                    f"Mismatched sizes between {self} table and ndarray of shape {other.shape}!"
                )
            psi_k = self.table.numpy().flatten()
            phi_k = other.flatten()
        else:
            if tuple(self.table.shape) != tuple(other.table.shape):
                raise ValueError(
                    f"Mismatched supports between {self} and {other}!"
                )
            psi_k = self.table.numpy().flatten()
            phi_k = other.table.numpy().flatten()

        psi_k /= psi_k.sum()
        phi_k /= psi_k.sum()
        return np.sum(
            psi_k * (np.log2(psi_k) - np.log2(phi_k))
        )

    def entropy(self,) -> float:
        """
        Compute this factor's entropy.

        The entropy for the factor :math:`\psi` is equal to

        .. math:: \mathrm{H}(\psi) = -\sum_x \\frac{\psi(x)}{Z_\psi} \log \\frac{\psi(x)}{Z_\psi}.

        The number :math:`Z_\psi` is the partition function that converts the factors into proper
        probability distributions. Note that the complexity of this method is exponential in the 
        degree of the factor. NOTE: The logarithm is taken in base 2; the result has units of bits.
        """
        psi_k = self.table.numpy().flatten()
        psi_k /= psi_k.sum()
        return np.sum(
            -1.0 * psi_k * np.log2(psi_k)
        )


class FactorGraph(abc.ABC):
    """
    Abstract base class from which all factor graphs must inherit.
    Factor graphs must implement one class method and
    three methods:

    + `learn`: class method that returns an instance of the factor graph,
        with initialized tables (i.e., tables that are non-null and probably
        have been learned from data)
    + `snapshot`: method that returns a copy of the factor graph
        without any linkage to global state of any kind (e.g., 
        param store, gradient tape)
    + `post_evidence`: allows evidence to be posted to the factor graph. Should
        return a factor graph with evidence posted to the factors. 
    + `query`: run queries against the graph. Currently the queries take the
        form of marginal clique probability distributions only.
    """

    @classmethod
    @abc.abstractmethod
    def learn(cls, *args, **kwargs):
        ...

    @abc.abstractmethod
    def snapshot(self,):
        ...

    @abc.abstractmethod
    def post_evidence(self, var, level):
        ...

    @abc.abstractmethod
    def query(self, variables,):
        ...


class DiscreteFactorGraph(FactorGraph):
    """
    A `DiscreteFactorGraph` is a collection of `DiscreteFactor`s which together constitute
    a bipartite graph linking variables to factors. The graph is represented only implicitly;
    no graph is ever constructed.
    """

    count = 0
    
    @classmethod
    def _next_id(cls):
        cls.count += 1
        return f"DiscreteFactorGraph{cls.count}"

    @classmethod
    def _mle_train(
        cls,
        fs2dim: collections.OrderedDict[str, tuple[int,...]],
        data: collections.OrderedDict[str, torch.Tensor],
        num_iterations: int=1000,
        lr: float=0.01,
        train_options: dict=dict(),
    ) -> torch.Tensor:
        """Trains the parameters of an MLE discrete factor model.
        """
        train_options = train_options or dict()
        svi = _setup_svi(discrete_factor_model, train_options=train_options,)
        pyro.clear_param_store()
        losses = torch.empty((num_iterations,))
        verbosity = train_options.get("verbosity", 100)
        num_iterations = train_options.get("num_iterations", num_iterations)
        
        for j in range(num_iterations):
            loss = svi.step(fs2dim, data=data,)
            if j % verbosity == 0:
                logging.info(f"On iteration {j}, -log p(x) = {loss}")
            losses[j] = loss
        return losses

    @classmethod
    def _transform_dims_data(
        cls,
        dimensions: Iterable[dimensions.FactorDimensions],
        data: pd.DataFrame,
        column_mapping: Mapping[str, str]=dict(),
    ):
        if len(column_mapping) > 0:
            data = data.rename(columns=column_mapping)
        variable_groups = [d.get_variables() for d in dimensions]
        dims = [d.get_dimensions() for d in dimensions]
        data = transform._df2od_torch(data, variable_groups, dims)
        fs2dim = collections.OrderedDict((d.get_factor_spec() for d in dimensions))
        return (fs2dim, data)

    @classmethod
    def _mle_train_from_generator(
        cls,
        dimensions: Iterable[dimensions.FactorDimensions],
        data: Callable,
        train_options: Optional[dict]=dict(),
        column_mapping: Mapping[str, str]=dict(),
    ):
        """
        Trains an MLE discrete factor model using a data generator. 

        `data` should be a callable that, when called with no arguments, emits a generator that
        emits `pd.DataFrame`s.

        """
        train_options = train_options or dict()
        num_epochs = train_options.get("num_epochs", 1)
        svi = _setup_svi(discrete_factor_model, train_options=train_options,)
        pyro.clear_param_store()
        losses = []
        verbosity = train_options.get("verbosity", 100)
        j = 0
        len_column_mapping = len(column_mapping)
        for epoch in range(num_epochs):
            logging.info(f"On epoch {epoch}")
            data_generator = data()
            for batch in data_generator:  # NOTE: assuming that each batch is a pd.DataFrame
                (fs2dim, batch) = cls._transform_dims_data(
                    dimensions,
                    batch,
                    column_mapping=column_mapping,
                )
                loss = svi.step(fs2dim, data=batch,)
                if j % verbosity == 0:
                    logging.info(f"On iteration {j}, -log p(x) = {loss}")
                losses.append(loss)
                j += 1
        return (torch.tensor(losses), fs2dim)

    @classmethod
    def learn(
        cls,
        dimensions: Iterable[dimensions.FactorDimensions],
        data: Union[Iterable, pd.DataFrame],
        train_options: Optional[dict]=None,
        column_mapping: Mapping[str, str]=dict(),
    ):
        """Learns parameters of factors in a factor graph from data.

        + `dimensions`: an iterable of `FactorDimensions`, each of which describes the variables
            related by that factor.
        + `data`: `pandas.DataFrame`. Each column must be equal length and correspond to observations
            of the variable given in the header of the column.
        """
        if type(data) == pd.DataFrame:
            (fs2dim, data) = cls._transform_dims_data(
                dimensions,
                data,
                column_mapping=column_mapping,
            )
            losses = cls._mle_train(
                fs2dim,
                data,
                train_options=train_options,
            )
        # TODO: test to ensure that the passed Iterable *is* a generator better
        else:  # is a generator
            (losses, fs2dim) = cls._mle_train_from_generator(
                dimensions,
                data,
                train_options=train_options,
                column_mapping=column_mapping,  # calls _transform_dims_data internally
            )
        factors = [
            DiscreteFactor(f"DiscreteFactor({fs})", fs, dim, pyro.param(make_factor_name(fs)))
            for (fs, dim) in fs2dim.items()
        ]
        new_factor_graph = cls(*factors, ts=None)
        new_factor_graph._learned = True
        assert new_factor_graph.fs2dim == fs2dim
        return (new_factor_graph, losses)

    def __init__(
        self,
        *factors: DiscreteFactor,
        ts: Optional[datetime.datetime]=None,
        inference_cache_size: int=1,
    ):
        self.ts = ts
        self.factors = collections.OrderedDict({
            f.name: f for f in factors
        })
        self.id = type(self)._next_id()
        # NOTE this is okay even if we overwrite, since two factors with the same fs must by definition
        # relate the same clique of variables, hence the same dimension tuple.
        self.fs2dim = collections.OrderedDict({
            f.fs: f.dim for f in self.factors.values()
        })

        self._evidence_cache = list()
        self._learned = False

        # potentially use cache during inference
        self.set_inference_cache_size(inference_cache_size)
        self._CACHED_INTERMEDIATES = None
        self.reset_inference_cache()

        # precompute elimination paths
        self.build_contract_expr()

    def build_contract_expr(
        self,
        result_spec: str="",
        optimize="branch-2",
    ):
        """
        Dispatches to `make_contract_expr`, hence uses LRU cache.

        Defaults: `result_spec=""` means that the path is optimized for
        computing the partition function. `optimize="branch-2"` uses the 
        branch-2 algorithm from `opt_einsum`, see that documentation for 
        details.
        """
        self._CONTRACT_EXPR = make_contract_expr(
            ",".join([f.fs for f in self.factors.values()]),
            result_spec,
            tuple(f.shape for f in self.factors.values()),
            optimize=optimize,
        )

    def reset_contract_expr(self,):
        """
        Resets the contraction expression to something "safe" -- i.e., calls
        `build_contract_expr` with defaults.
        """
        self.build_contract_expr()

    def reset_inference_cache(self,):
        """
        Garbage-collects the intermedates cache, initializes a new one, and
        sets the inference iterations to zero. 
        """
        if self._CACHED_INTERMEDIATES is not None:
            del self._CACHED_INTERMEDIATES
        with opt_einsum.shared_intermediates() as cache: pass
        self._CACHED_INTERMEDIATES = cache
        self._inference_iterations = 0

    def set_inference_cache_size(self, size: int):
        """
        Sets the inference cache size to a positive integer.
        """
        if size < 1:
            raise ValueError("Size of inference cache must be >= 1!")
        self._INFERENCE_CACHE_SIZE = size

    def get_inference_cache_size(self,):
        """
        Returns the inference cache size.
        """
        return self._INFERENCE_CACHE_SIZE

    def snapshot(self,):
        """
        Snapshots the state of a factor graph. Returns 
        a new factor graph with values identical to the original
        and a `.ts` attribute indicating a microsecond-level timestamp
        of when the snapshot was taken.
        """
        return DiscreteFactorGraph(
            *(f.snapshot() for f in self.factors.values()),
            ts=datetime.datetime.now()
        )

    def __repr__(self,):
        s = f"DiscreteFactorGraph(id={self.id}\n"
        for f in self.factors.values():
            s += f"\t{f},\n"
        if self.ts is not None:
            s += f"\tts={self.ts}\n"
        s +=")"
        return s

    def get_shapes(self,):
        """Returns dimensions of each factor in a `collections.OrderedDict`
        """
        return collections.OrderedDict({
            factor_name: factor.shape for (factor_name, factor) in self.factors.items()
        })

    def get_factor(self, name: str):
        """Returns the factor corresponding to the factor name. Raises `KeyError` if this
        factor is not in the graph.
        """
        return self.factors[name].get_factor()

    def post_evidence(self, var: str, level: int):
        """Posts evidence to the factor graph. 

        Calls `._post_evidence(var, level)` on each factor that
        contains `var`; see documentation of `._post_evidence(...)`
        in `DiscreteFactor`. 
        """
        ev = (var, level)
        if ev not in self._evidence_cache:
            # NOTE: okay after changing fs -> name since this expresses correct intent:
            # looking up factor by *unique* name, applying evidence, and setting the value
            for name in self.factors.keys():
                self.factors[name] = self.factors[name]._post_evidence(var, level)
            self._evidence_cache.append(ev)
            self.fs2dim[name] = self.factors[name].shape
        else:
            raise ValueError(f"Already posted evidence {ev}.") 

    def query(
        self,
        variables: str,
        safe: bool=True,
    ):
        """Queries the factor graph for marginal or posterior distribution 
        corresponding to `variables`. Returns a `DiscreteFactor` instance. 

        Interpretation by example: suppose the factor graph has two factors, 
        `ab` and `bc`. Passing `variables = "b"` to this method computes
        :math:`p(b)`. If evidence has first been applied, e.g., by calling 
        `.post_evidence("c", 2)`, then this method returns :math:`p(b | c = 2)`. 
        Passing `variables = "ab" after calling `.post_evidence("c", 2)` 
        would return :math:`p(a, b | c = 2)`, and so on.

        The optional argument `safe` determines whether or not to use a pre-computed
        contraction path. If `not safe`, then the instance's pre-computed contraction
        expression (itself computed using `.build_contract_expr(...)` is used to compute 
        the query, It is *not checked* if this contraction expression is even valid, nor 
        is it checked that this contraction expression is fast. If in doubt, use `safe = True`,
        the default value.
        """
        # TODO: should we call network_string on initialization and then
        # update only if we add / remove factors from network?
        # this adds to query time as is. 
        network_string = ",".join(self.fs2dim.keys())
        if safe:
             self.build_contract_expr(result_spec=variables,)
        with opt_einsum.shared_intermediates(self._CACHED_INTERMEDIATES):
            with torch.no_grad():
                result_table = discrete_marginal(
                    self._CONTRACT_EXPR,
                    # NOTE: have to pass explicitly since we copy and alter tensors
                    # taken from param store
                    *(factor.table for factor in self.factors.values()),
                )
        self._inference_iterations += 1
        if self._inference_iterations >= self._INFERENCE_CACHE_SIZE:
            self.reset_inference_cache()
        return DiscreteFactor(
            f"{self.id}-query={variables}",
            variables,
            result_table.shape,
            result_table
        )

    def link(self, other):
        """
        Join this factor graph with another of the same type. Raises `ValueError` if there are
        two factors with the same name. Returns another DiscreteFactorGraph that is the
        (disjoint) union of the two original factor graphs.

        All factors are snapshot so there is no linkage to any global state in the resulting graph.
        """
        for (my_name, my_factor) in self.factors.items():
            for (their_name, their_factor) in other.factors.items():
                if my_name == their_name:
                    raise ValueError(f"Can't link because {self.id} and {other.id} both have factors named {my_name}")
        return DiscreteFactorGraph(
            *([f.snapshot() for f in self.factors.values()] + [f.snapshot() for f in other.factors.values()])
        )
