
import itertools
from typing import Literal, Union

import torch

from . import dimensions, factor


def _get_pairwise_constraint_table(
    dim: tuple[int, int],
    relation: Literal["==", "!=", "<", ">", "<=", ">="]
) -> torch.Tensor:
    """
    Create tensor representing the pairwise relation constraint.
    """
    if relation == "==":
        the_tensor = torch.zeros(dim)
        # the_tensor[
        #     range(len(the_tensor)),
        #     range(len(the_tensor))
        # ] = torch.tensor(1.0)
        the_tensor.fill_diagonal_(1.0)
    elif relation == "!=":
        the_tensor = torch.ones(dim)
        # the_tensor[
        #     range(len(the_tensor)),
        #     range(len(the_tensor))
        # ] = torch.tensor(0.0)
        the_tensor.fill_diagonal_(0.0)
    elif relation == "<":
        the_tensor = torch.ones(dim).triu(diagonal=1,)
    elif relation == ">":
        the_tensor = torch.ones(dim).tril(diagonal=-1,)
    elif relation == "<=":
        the_tensor = torch.ones(dim).triu()
    elif relation == ">=":
        the_tensor = torch.ones(dim).tril()
    return the_tensor


def _to_constraint_value(x: torch.Tensor) -> torch.Tensor:
    """
    Turns the :math:`[0, \infty)`-valued tensor into a 
    :math:`\{0, 1\}`-valued tensor.
    """
    return torch.where(
        x > torch.tensor(0.0),
        torch.tensor(1),
        torch.tensor(0)
    )


class ConstraintFactor(factor.DiscreteFactor):
    """
    Factors that express constraints between variables.
    If :math:`\psi(x) > 0`, that means that the allocation :math:`x`
    is allowed by the constraint; if :math:`\psi(x) = 0`, the allocation
    is not allowed by the constraint.

    Note that construction of this class requires either calling a classmethod
    or passing a :math:`\{0,1\}`-valued tensor. However, the result of running
    variable elimination on a `DiscreteFactorGraph` containing `ConstraintFactors`
    may return a factor with :math:`[0,\infty)`-valued elements.
    """

    @classmethod
    def random(
        cls,
        dim: dimensions.FactorDimensions
    ):  
        """
        Construct a random constraint factor. If the table is defined on the support
        :math:`S = \prod_{1\leq n \leq N}\{0,1\}^{D_n}` and each :math:`D_n \in \{1, 2, ...\}`,
        then the pmf of the table is given by :math:`p(\psi) = 2^{-\prod_{1\leq n \leq N} D_n}`.
        """
        the_fs = dim.get_variable_str()
        the_dim = dim.get_dimensions()
        the_table = torch.randint(0, 1 + 1, the_dim)
        return cls(the_fs, the_dim, the_table, factor_name_prefix="RANDOM_CONSTRAINT")

    @classmethod
    def value(
        cls,
        dim: dimensions.VariableDimensions,
        low: int,
        high: int
    ):
        """
        Defines a value constraint on a variable; restricts the range of the variable
        to :math:`\{\mathrm{low},...,\mathrm{high} - 1\}`.
        """
        if len(dim) > 1:
            raise ValueError("Can define value constraint only on 1d factor!")
        the_fs = dim.get_variable_str()
        the_dim = dim.get_dimensions()
        the_table = torch.zeros(the_dim)
        the_table[low:high] = torch.tensor(1.0)
        return cls(the_fs, the_dim, the_table, factor_name_prefix="VALUE_CONSTRAINT")

    @classmethod
    def pairwise(
        cls,
        dim: dimensions.FactorDimensions,
        relation: Literal["==", "!=", "<", ">", "<=", ">="],
        factor_name_prefix: str="PAIRWISE",
    ):
        """
        Define common pairwise constraints for two variables. 

        Suppose that :math:`a` and :math:`b` are variables taking values in
        :math:`\{0,...,D_a\}` and :math:`\{0,...,D_b\}` respectively. 
        By passing the appropriate `relation` argument, this factor can express
        :math:`a = b`, :math:`a \\neq b`, :math:`a < b`, :math:`a \leq b`, 
        :math:`a > b`, or :math:`a \geq b`.
        """
        if len(dim) != 2:
            raise ValueError(".pairwise(...) works only with exactly two variables!")
        the_fs = dim.get_variable_str()
        the_dim = dim.get_dimensions()
        the_table = _get_pairwise_constraint_table(the_dim, relation)
        return cls(the_fs, the_dim, the_table, factor_name_prefix=factor_name_prefix)

    def __init__(
        self,
        fs: str,
        dim: Union[torch.Size, tuple[int,...]],
        table: torch.Tensor,
        factor_name_prefix: str="CONSTRAINT",
    ):
        # check that all entries in the table are zero or one
        if not all(map(lambda x: (x == 0) | (x == 1), table.view((-1,)))):
            raise ValueError("Construct ConstraintFactor with only {0,1} tensor.")
        name = f"ConstraintFactor({fs})"
        super().__init__(
            name, fs, dim, table, factor_name_prefix=factor_name_prefix,
        )


def all_different(
    *variables: dimensions.VariableDimensions
) -> tuple[ConstraintFactor,...]:
    """
    Enforces the constraint that values of all passed variables must be different.

    If :math:`N \geq 2` is the number of variables for which the constraint must hold,
    this function returns a tuple of :math:`N(N - 1)/2` ``ConstraintFactors``.
    """
    return (
        ConstraintFactor.pairwise(
            dimensions.FactorDimensions(x, y),
            "!=",
            factor_name_prefix="ALL_DIFFERENT",
        )
        for (x, y) in itertools.combinations(variables, 2)
    )


def all_equal(
    *variables: dimensions.VariableDimensions
) -> tuple[ConstraintFactor,...]:
    """
    Enforces the constraint that values of all passed variables must be equal.

    If :math:`N \geq 2` is the number of variables for which the constraint must hold,
    this function returns a tuple of :math:`N(N - 1)/2` ``ConstraintFactors``.
    """
    return (
        ConstraintFactor.pairwise(
            dimensions.FactorDimensions(x, y),
            "==",
            factor_name_prefix="ALL_EQUAL",
        )
        for (x, y) in itertools.combinations(variables, 2)
    )


def enumerate_feasible(f: ConstraintFactor) -> torch.Tensor:
    """
    Returns a tensor of (marginal) feasible allocations. The interpretation
    of this tensor depends on the query that was originally passed to 
    the factor graph.
    """
    return torch.nonzero(_to_constraint_value(f.table))
    