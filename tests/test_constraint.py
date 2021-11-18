
import logging

import pytest
import torch
from models import dimensions, factor


@pytest.mark.constraint
def test_constraint_create():
    d1, d2 = 10, 12
    dimgen = dimensions.DimensionsFactory("var1", "var2")
    dimgen("var1", d1)
    dimgen("var2", d2)
    constraint_dim = dimgen(("var1", "var2"))

    random_constraint = factor.ConstraintFactor.random(constraint_dim)
    assert random_constraint.table.shape == (d1, d2)
    logging.info(f"Created random constraint factor: {random_constraint}")


@pytest.mark.constraint
def test_pairwise_constraint_create():
    d1, d2 = 3, 3  # minimally interesting
    dimgen = dimensions.DimensionsFactory("var1", "var2")
    dimgen("var1", d1)
    dimgen("var2", d2)
    constraint_dim = dimgen(("var1", "var2"))

    equality_pairwise = factor.ConstraintFactor.pairwise(constraint_dim, "==")
    logging.info(
        f"Generated pairwise equality constraint table:\n{equality_pairwise.table}"
    )
    assert torch.equal(
        equality_pairwise.table,
        torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    )
    inequality_pairwise = factor.ConstraintFactor.pairwise(constraint_dim, "!=")
    logging.info(
        f"Generated pairwise inequality constraint table:\n{inequality_pairwise.table}"
    )
    assert torch.equal(
        inequality_pairwise.table,
        torch.tensor([[0., 1., 1.], [1., 0., 1.], [1., 1., 0.]])
    )
    le_pairwise = factor.ConstraintFactor.pairwise(constraint_dim, "<")
    logging.info(
        f"Generated pairwise < constraint table:\n{le_pairwise.table}"
    )
    assert torch.equal(
        le_pairwise.table,
        torch.tensor([[0., 1., 1.], [0., 0., 1.], [0., 0., 0.]])
    )
    ge_pairwise = factor.ConstraintFactor.pairwise(constraint_dim, ">")
    logging.info(
        f"Generated pairwise > constraint table:\n{ge_pairwise.table}"
    )
    assert torch.equal(
        ge_pairwise.table,
        torch.tensor([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.]])
    )
    leq_pairwise = factor.ConstraintFactor.pairwise(constraint_dim, "<=")
    logging.info(
        f"Generated pairwise <= constraint table:\n{leq_pairwise.table}"
    )
    assert torch.equal(
        leq_pairwise.table,
        torch.tensor([[1., 1., 1.], [0., 1., 1.], [0., 0., 1.]])
    )
    geq_pairwise = factor.ConstraintFactor.pairwise(constraint_dim, ">=")
    logging.info(
        f"Generated pairwise >= constraint table:\n{geq_pairwise.table}"
    )
    assert torch.equal(
        geq_pairwise.table,
        torch.tensor([[1., 0., 0.], [1., 1., 0.], [1., 1., 1.]])
    )
