"""
Factor graphs
*************

This module contains functions and classes related to factor graphs. Factor graphs are undirected
bipartite graphs that relate variable nodes, which are quantities of interest such as the 
genre into which a movie may be classified or the price of a limit bid order, to factor
nodes, which are non-negative functions that relate values of variables to one another.
If a factor graph has :math:`N` variables :math:`x_1,...,x_N`, the factor graph can 
be expressed as

.. math:: p(x_1,...,x_N) = \\frac{1}{Z(\\theta)} 
    \prod_{\\mathrm{cl}}\psi_{\mathrm{cl}}(x_{\mathrm{cl}}; \\theta)
    :label: fg-eq

The notation :math:`x_\mathrm{cl}` denotes a clique of variables that are related by the factor
:math:`\psi_\mathrm{cl}`. For example, a factor graph that relates all pairs of nodes would have
joint density given by

.. math:: p(x_1,...,x_N) = \\frac{1}{Z(\\theta)} 
    \prod_{1\leq i < j \leq N}\psi_{i,j}(x_i, x_j; \\theta)

The vector :math:`\\theta` denotes the vector of parameters for the entire graph (note that
"vector" in this context can be interpreted as "all tensors of parameters"; you could create a
vector from all tensors of parameters by `squeeze`-ing each of them and `concat`-enating them 
together). The value of a factor is given by
:math:`\psi_{\mathrm{cl}}(x_{\mathrm{cl}}; \\theta) = \\theta_{x_\mathrm{cl}}`, the corresponding
element in the :math:`|\mathrm{cl}|` dimensional tensor of values.

Values of a factor denote how likely it is that variable elements co-occur. This module is concerned
with two factor interpretations: probability factors, which denote un-normalized probablility that
variable elements co-occur; and constraint factors, which are :math:`\{0,1\}`-valued tensors that
denote whether it is possible or not for variable elements to co-occur.
The number :math:`Z(\\theta)` in Eq. :eq:`fg-eq` is a normalization constant. In the case of
a factor graph containing only constraint nodes its value is immaterial (since a configuration of nodes
is valid iff :math:`p(x_1,...,x_N)` is nonzero), but in the case of a factor graph representing a 
probability mass function (pmf) it is the normalizer, or the number that makes :math:`p(x_1,...,x_N)`
a valid pmf. In this case, the partition function is equal to

.. math:: Z(\\theta) = \sum_{x_1,...,x_N}
    \prod_{\\mathrm{cl}}\psi_{\mathrm{cl}}(x_{\mathrm{cl}}; \\theta)

A naive computation of this quantity is exponential in the number of variables in the factor graph.
Exploiting the structure of the factor graph during computation lowers the complexity to only
exponential in the size of the highest-degree factor, which may be a significant reduction in complexity
(e.g., for a factor graph with thousands of variables but only pairwise relationships between variables).

"""

import abc
import collections
import datetime
import logging
from typing import Callable, Optional, Union

import mypy
import pyro
import pyro.distributions as dist
import pyro.distributions.constraints as constraints
from pyro.ops.contract import einsum
import torch


def discrete_joint_conditioned(eq: str, *tensors: torch.Tensor):
    """
    Computes the distribution :math:`p(V, E = e)` via (cached) variable elimination. 

    Wrapper around `pyro.ops.contract.einsum` that does not enforce normalization to 
    a pmf (since that occurs in `discrete_marginal`).

    TODO: ensure that we are using the most optimal caching strategy internally. This 
    may involve actually using the underlying opt_einsum function call instead so that we can 
    specify what tensors remain constant and other optimizations.
    """
    return einsum(eq, *tensors, modulo_total=True)[0]


def discrete_marginal(eq: str, *tensors: torch.Tensor):
    """
    Computes the exact discrete_marginal distribution via (cached) variable elimination.

    This method dispatches to `discrete_joint_conditioned` which computes 
    :math:`p(V, E = e)`. This method then computes
    :math:`p(V | E = e) = p(V, E = e) / p(E = e)`. Note that the complexity of this method
    scales as :math:`\mathcal{O}(D^{|V|})` where :math:`D` is the maximum variable
    support cardinality.
    """
    unscaled = discrete_joint_conditioned(eq, *tensors)
    return (unscaled / torch.sum(unscaled))


def make_factor_name(fs: str):
    """
    Makes a factor name from a dimension specification string. 
    
    NOTE: right now this name is *not* unique. 
    
    TODO: make this name unique. 
    """
    return f"f_{fs}"


def discrete_factor_model(
    fs2dim: collections.OrderedDict[str,tuple[int,...]],
    data: Optional[collections.OrderedDict[str,torch.Tensor]]=None,
    query_var: Optional[str]=None
) -> Optional[torch.Tensor]:
    """A Pyro model corresponding to a discrete factor graph. This model supports both MLE parameter
    learning and inference.

    This function takes a `collections.OrderedDict` of {dimension name string: dimension size tuple}, e.g., 
    {"ab": (2, 3), "bc": (3, 4)}. 
    
    NOTE: this functionality is suboptimal because it does not allow for 
    multiple factors that relate the same dimensions (e.g., one that maps probabilities and another that 
    maps constraints). 
    
    TODO: this must be fixed. 

    If `data` is not None, then this function scores the observed data against the current values of the factors
    using Pyro machiner. 
    
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
            make_factor_name(fs),
            torch.ones(dim),
            constraint=constraints.positive
        )
    network_string = ",".join(fs2dim.keys())

    if not query_var:
        for var in fs2dim.keys():  # iterate through all defined cliques
            pr = discrete_marginal(f"{network_string}->{var}", *factors.values())
            with pyro.plate(f"{var}-plate") as ix:
                pyro.sample(
                    f"{var}-data",
                    dist.Categorical(pr.reshape((-1,))),
                    obs=data[var]
                )
    else:
        with torch.no_grad():
            return discrete_marginal(f"{network_string}->{query_var}", *factors.values())

def mle_train(
    model: Callable,
    model_args: tuple,
    model_kwargs: dict,
    num_iterations: int=1000,
    lr: float=0.01,
) -> torch.Tensor:
    """Trains the parameters of an MLE model. 
    
    NOTE: the model must actually be an MLE model 
    (i.e., have no latent random variables) as this function maximizes the ELBO using an empty 
    guide, which will result in an error if the model has latent random variables.

    The callable model must be a Pyro stochastic function, while the model_args and model_kwargs are the 
    positional and keyword arguments that the model requires. The optimization will proceed for `num_iterations`
    iterations using the learning rate `lr`. 
    
    NOTE: right now this uses the Adam optimizer. We should a) allow the user
    to specify what optimizers they want to use and b) experiment with choices of optimizer on real problems to 
    see if we can find heuristics on which ones are better choices conditioned on context. 
    """
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
        self.name = name
        self.fs = fs
        self.dim = dim
        if (table is not None) and (self.dim != table.shape):
            raise ValueError(
            f"Dims {dim} don't match table shape {table.shape}"
        )
        self.table = table
        self.shape = None if self.table is None else self.table.shape

        self._variables = [var for var in self.fs]
        self._variables_to_axis = collections.OrderedDict({
            var: i for (i, var) in enumerate(self._variables)
        })

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
        return pyro.param(make_factor_name(self.fs))
    
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
    def learn(
        cls,
        fs2dim: collections.OrderedDict[str, tuple[int,...]],
        data: collections.OrderedDict[str, torch.Tensor],
    ):
        """Learns parameters of factors in a factor graph from data.

        + `fs2dim`: `collections.OrderedDict` of {variable specification string: tuple of dimension sizes}.
            For example, if variable `a` had 4 levels and variable `b` had 7 levels, `fs2dim` would have
            an entry `"ab": (4, 7)`. 
        + `data`: `collections.OrderedDict` of {variable specification string: observations tensor}. 
            All observations tensors must be the same length; their length is equal to the number of 
            observations of model state. The observations tensors must be 1d. The numbers in them 
            correspond to the index that would occur when the multidimensional index corresponding 
            to an observation across multiple variables in the factor was flattened.

            TODO: make this description less confusing

            TODO: refactor how data is represented, should instead be with a pandas.DataFrame or similar
        """
        losses = mle_train(
            discrete_factor_model,
            (fs2dim,),
            dict(data=data),
        )
        factors = [
            DiscreteFactor(f"DiscreteFactor({fs})", fs, dim, pyro.param(make_factor_name(fs)))
            for (fs, dim) in fs2dim.items()
        ]
        new_factor_graph = cls(*factors, ts=None)
        assert new_factor_graph.fs2dim == fs2dim
        return (new_factor_graph, losses)

    def __init__(
        self,
        *factors: DiscreteFactor,
        ts: Optional[datetime.datetime]=None
    ):
        self.ts = ts
        self.factors = collections.OrderedDict({
            f.fs: f for f in factors
        })
        self.id = type(self)._next_id()
        self.fs2dim = collections.OrderedDict({
            f.fs: f.dim for f in self.factors.values()
        })

        self._evidence_cache = list()

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
            ix: f.shape for (ix, f) in self.factors.items()
        })

    def get_factor(self, fs: str):
        """Returns the factor corresponding to the dimension string
        `fs`. 
        
        TODO: this should be updated to return factor by name
        instead of by dimension string; right now this has the implicit
        assumption that each dimension string is unique, which does not
        have to be the case.
        """
        return self.factors[fs].get_factor()

    def post_evidence(self, var: str, level: int):
        """Posts evidence to the factor graph. 

        Calls `._post_evidence(var, level)` on each factor that
        contains `var`; see documentation of `._post_evidence(...)`
        in `DiscreteFactor`. 
        """
        ev = (var, level)
        if ev not in self._evidence_cache:
            for name in self.factors.keys():
                self.factors[name] = self.factors[name]._post_evidence(var, level)
            self._evidence_cache.append(ev)
            self.fs2dim[name] = self.factors[name].shape
        else:
            raise ValueError(f"Already posted evidence {ev}.") 

    def query(self, variables: str):
        """Queries the factor graph for marginal or posterior distribution 
        corresponding to `variables`. Returns a `DiscreteFactor` instance. 

        Interpretation by example: suppose the factor graph has two factors, 
        `ab` and `bc`. Passing `variables = "b"` to this method computes
        :math:`p(b)`. If evidence has first been applied, e.g., by calling 
        `.post_evidence("c", 2)`, then this method returns :math:`p(b | c = 2)`. 
        Passing `variables = "ab" after calling `.post_evidence("c", 2)` 
        would return :math:`p(a, b | c = 2)`, and so on.
        """
        # TODO: should we call network_string on initialization and then
        # update only if we add / remove factors from network?
        # this adds to query time as is. 
        network_string = ",".join(self.fs2dim.keys())
        # NOTE: have to pass explicitly since we copy and alter tensors
        # taken from param store
        with torch.no_grad():
            result_table = discrete_marginal(
                f"{network_string}->{variables}",
                *(DiscreteFactor.table for DiscreteFactor in self.factors.values())
            )
        return DiscreteFactor(
            f"{self.id}-query={variables}",
            variables,
            result_table.shape,
            result_table
        )
