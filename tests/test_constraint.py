
import logging

import pytest
import torch
from models import (
    constraint,
    dimensions,
    factor,
    transform
)


@pytest.mark.constraint
def test_constraint_create():
    d1, d2 = 10, 12
    dimgen = dimensions.DimensionsFactory("var1", "var2")
    dimgen("var1", d1)
    dimgen("var2", d2)
    constraint_dim = dimgen(("var1", "var2"))

    random_constraint = constraint.ConstraintFactor.random(constraint_dim)
    assert random_constraint.table.shape == (d1, d2)
    logging.info(f"Created random constraint factor: {random_constraint}")

    var1_dim = dimgen(("var1",))
    low = 3
    high = 8
    level_constraint = constraint.ConstraintFactor.value(var1_dim, low, high)
    assert torch.equal(
        level_constraint.table,
        torch.tensor([0., 0., 0., 1., 1., 1., 1., 1., 0., 0.])
    )


@pytest.mark.constraint
def test_all_eq_and_not_eq():
    d1, d2, d3 = 2, 3, 4
    dimgen = dimensions.DimensionsFactory("a", "b", "c")
    dimgen("a", d1)
    dimgen("b", d2)
    dimgen("c", d3)

    all_diff_factors = constraint.all_different(
        *(dimgen.get_variable(var) for var in ["a", "b", "c"])
    )
    all_diff_graph = factor.DiscreteFactorGraph(*all_diff_factors)
    all_eq_factors = constraint.all_equal(
        *(dimgen.get_variable(var) for var in ["a", "b", "c"])
    )
    all_eq_graph = factor.DiscreteFactorGraph(*all_eq_factors)
    # actually compute all feasible worlds -- this operation has high complexity
    # and we wouldn't do this on larger problems
    all_diff_worlds = all_diff_graph.query("abc")
    all_diff_worlds = constraint.enumerate_feasible(all_diff_worlds)
    logging.info(f"All diff worlds:\n{all_diff_worlds}")
    all_eq_worlds = all_eq_graph.query("abc")
    all_eq_worlds = constraint.enumerate_feasible(all_eq_worlds)
    logging.info(f"All eq worlds:\n{all_eq_worlds}")


@pytest.mark.constraint
def test_pairwise_constraint_create():
    d1, d2 = 3, 3  # minimally interesting
    dimgen = dimensions.DimensionsFactory("var1", "var2")
    dimgen("var1", d1)
    dimgen("var2", d2)
    constraint_dim = dimgen(("var1", "var2"))

    equality_pairwise = constraint.ConstraintFactor.pairwise(constraint_dim, "==")
    logging.info(
        f"Generated pairwise equality constraint table:\n{equality_pairwise.table}"
    )
    assert torch.equal(
        equality_pairwise.table,
        torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    )
    inequality_pairwise = constraint.ConstraintFactor.pairwise(constraint_dim, "!=")
    logging.info(
        f"Generated pairwise inequality constraint table:\n{inequality_pairwise.table}"
    )
    assert torch.equal(
        inequality_pairwise.table,
        torch.tensor([[0., 1., 1.], [1., 0., 1.], [1., 1., 0.]])
    )
    le_pairwise = constraint.ConstraintFactor.pairwise(constraint_dim, "<")
    logging.info(
        f"Generated pairwise < constraint table:\n{le_pairwise.table}"
    )
    assert torch.equal(
        le_pairwise.table,
        torch.tensor([[0., 1., 1.], [0., 0., 1.], [0., 0., 0.]])
    )
    ge_pairwise = constraint.ConstraintFactor.pairwise(constraint_dim, ">")
    logging.info(
        f"Generated pairwise > constraint table:\n{ge_pairwise.table}"
    )
    assert torch.equal(
        ge_pairwise.table,
        torch.tensor([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.]])
    )
    leq_pairwise = constraint.ConstraintFactor.pairwise(constraint_dim, "<=")
    logging.info(
        f"Generated pairwise <= constraint table:\n{leq_pairwise.table}"
    )
    assert torch.equal(
        leq_pairwise.table,
        torch.tensor([[1., 1., 1.], [0., 1., 1.], [0., 0., 1.]])
    )
    geq_pairwise = constraint.ConstraintFactor.pairwise(constraint_dim, ">=")
    logging.info(
        f"Generated pairwise >= constraint table:\n{geq_pairwise.table}"
    )
    assert torch.equal(
        geq_pairwise.table,
        torch.tensor([[1., 0., 0.], [1., 1., 0.], [1., 1., 1.]])
    )


@pytest.mark.slow
@pytest.mark.constraint
def test_arc_csp_solve():
    # Set up and solve a basic CSP using variable elimination
    # First, let's define the variables and their domains
    # Variables are defined on the discrete set {0, ..., dim - 1}
    variables = ["a", "b", "c", "d"]
    dims = [3, 3, 10, 20]
    dimgen = dimensions.DimensionsFactory(*variables)

    for (var, dim) in zip(variables, dims):
        dimgen(var, dim)

    # Now, let's describe the constraints on the variables.
    # We encode these constraints in factors.

    # First, a and b must be different and so must c and d
    ab_neq = constraint.ConstraintFactor.pairwise(dimgen(("a", "b")), "!=")
    cd_neq = constraint.ConstraintFactor.pairwise(dimgen(("c", "d")), "!=")

    # Second, c > a and d > b
    ca_ge = constraint.ConstraintFactor.pairwise(dimgen(("c", "a")), ">")
    db_ge = constraint.ConstraintFactor.pairwise(dimgen(("d", "b")), ">")

    # Now construct a constraint graph
    constraint_graph = factor.DiscreteFactorGraph(
        ab_neq, cd_neq, ca_ge, db_ge
    )
    logging.info(f"Created constraint graph:\n{constraint_graph}")

    # We can ask all kinds of questions about feasible configuration of variables!

    # In this context, univariate marginal query p(X=x) translates to "Are there
    # *any* state configurations for which X = x is feasible?"
    for var in variables:
        univariate_marginal = constraint_graph.query(var)
        logging.info(f"Marginal feasibility for {var}:\n{constraint._to_constraint_value(univariate_marginal.table)}")

    # As expected, we see that for almost every value of each of the variables there exists
    # some configuration for which that value is feasible

    # Similarly, we can check clique-wise constraints. Let's see for which value of
    # (a, d) there exists a configuration for which that value is feasible
    ab_marginal = constraint_graph.query("ad")
    logging.info(f"Marginal feasibility for (a, d):\n{constraint._to_constraint_value(ab_marginal.table)}")

    # Great, so there are some feasible configurations. Let's find the valid (b, c) 
    # if we suppose we want to set a = 2, d = 10
    # We'll snapshot the graph so we don't modify the original constraint graph
    conditioned_constraint_graph = constraint_graph.snapshot()
    conditioned_constraint_graph.post_evidence("a", 2)
    conditioned_constraint_graph.post_evidence("d", 10)
    bc_given_ad_marginal = conditioned_constraint_graph.query("bc")
    logging.info(f"Marginal feasibility for (b, c):\n{constraint._to_constraint_value(bc_given_ad_marginal.table)}")
    
    # we see that variable b must be in {0, 1} (i.e., b = 2 is infeasible) and any value of
    # c >= 3 is feasible.
