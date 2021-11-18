import collections
import datetime
import logging
import time

import numpy as np
import pandas as pd
import pytest
import torch

import models


def get_data():
    return pd.DataFrame({
        "a": [0, 0, 1, 0, 1],
        "b": [0, 1, 2, 0, 1],
        "c": [1, 0, 3, 2, 1]
    })


def to_micro(t0, t1):
    return round(1e6 * (t1 - t0), 3)


@pytest.mark.slow
@pytest.mark.training
def test_snapshot():
    a_dim = models.dimensions.VariableDimensions("a", 2)
    b_dim = models.dimensions.VariableDimensions("b", 3)
    c_dim = models.dimensions.VariableDimensions("c", 4)

    ab_dim = models.dimensions.FactorDimensions(a_dim, b_dim)
    bc_dim = models.dimensions.FactorDimensions(b_dim, c_dim)
    ca_dim = models.dimensions.FactorDimensions(c_dim, a_dim)

    data = get_data()
    old_fg, losses_from_training = models.factor.DiscreteFactorGraph.learn(
        (ab_dim, bc_dim, ca_dim),
        data
    )
    logging.info(f"Learned a factor graph: {old_fg}")
    # snapshot allows us to tag state of a factor graph even if we
    # repeatedly update its weights. Don't just call deepcoppy due to
    # required errorchecks and detaching gradients
    new_fg = old_fg.snapshot()
    logging.info(f"Created a new factor graph via snapshot: {new_fg}")

    for (old_f, new_f) in zip(old_fg.factors.values(), new_fg.factors.values()):
        assert torch.equal(old_f.get_table(), new_f.get_table())

    # modify original graph and check that the new one doesn't change
    old_fg.factors["ab"].table[0, 1] += torch.tensor(1.0)
    assert not torch.equal(
        old_fg.factors["ab"].table,
        new_fg.factors["ab"].table
    )


@pytest.mark.slow
@pytest.mark.training
def test_integration_1():
    #fs2dim = get_fs2dim(dims())
    a_dim = models.dimensions.VariableDimensions("a", 2)
    b_dim = models.dimensions.VariableDimensions("b", 3)
    c_dim = models.dimensions.VariableDimensions("c", 4)

    ab_dim = models.dimensions.FactorDimensions(a_dim, b_dim)
    bc_dim = models.dimensions.FactorDimensions(b_dim, c_dim)
    ca_dim = models.dimensions.FactorDimensions(c_dim, a_dim)

    data = get_data()
    factor_graph, losses_from_training = models.factor.DiscreteFactorGraph.learn(
        (ab_dim, bc_dim, ca_dim),
        data
    )
    logging.info(f"Learned a factor graph: {factor_graph}")

    for variable in ["a", "b", "c"]:
        marginal = factor_graph.query(variable)
        logging.info(f"Marginal of {variable}: {marginal}")
        logging.info(f"p({variable})= {marginal.get_table()}")
    
    # look at the actual learned factors
    # not the same as the marginals of the cliques
    clique = "ab"
    p_clique = factor_graph.query(clique).get_table()
    clique_factor = factor_graph.get_factor(clique)
    logging.info(f"f_{clique} = {clique_factor}")
    logging.info(f"p({clique}) = {p_clique}")
    with torch.no_grad():
        normalized_clique_factor = clique_factor / torch.sum(clique_factor)
    logging.info(f"Normalized clique factor = {normalized_clique_factor}")
    # other marginal effects matter!
    assert not torch.equal(normalized_clique_factor, p_clique)  

    # note time to do three inference calculations as well
    # check out difference between observing 0, 1 and 2 variables
    logging.info(f"Factor graph shapes: {factor_graph.get_shapes()}")
    t0 = time.time()
    p_a = factor_graph.query("a").get_table()
    t1 = time.time()
    logging.info(f"Before posting evidence, p(a) = {p_a}")
    logging.info(f"Took {to_micro(t0, t1)}us to compute p(a)")

    logging.info(f"Posting b evidence")
    t0 = time.time()
    factor_graph.post_evidence("b", 2)
    p_a_cond_b = factor_graph.query("a").get_table()
    t1 = time.time()
    logging.info(f"Factor graph shapes: {factor_graph.get_shapes()}")
    logging.info(f"After posting b evidence, p(a) = {p_a_cond_b}")
    logging.info(f"Took {to_micro(t0, t1)}us to post b evidence and compute p(a|b)")

    # NOTE: factor graph *is stateful* so b evidence has already been posted!!!
    logging.info("Posting c evidence")
    t0 = time.time()
    factor_graph.post_evidence("c", 2)
    p_a_cond_bc = factor_graph.query("a").get_table()
    t1 = time.time()
    logging.info(f"Factor graph shapes: {factor_graph.get_shapes()}")
    logging.info(f"After posting b and c evidence, p(a) = {p_a_cond_bc}")
    logging.info(f"Took {to_micro(t0, t1)}us to post c evidence and compute p(a|b,c)")


@pytest.mark.slow
@pytest.mark.training
def test_visualization_and_divergence():
    a_dim = models.dimensions.VariableDimensions("a", 2)
    b_dim = models.dimensions.VariableDimensions("b", 3)
    c_dim = models.dimensions.VariableDimensions("c", 4)

    ab_dim = models.dimensions.FactorDimensions(a_dim, b_dim)
    bc_dim = models.dimensions.FactorDimensions(b_dim, c_dim)
    ca_dim = models.dimensions.FactorDimensions(c_dim, a_dim)

    data = pd.DataFrame({
        "a": [0, 0, 1, 0, 1],  # / 2
        "b": [0, 1, 2, 0, 1],  # / 3
        "c": [1, 0, 3, 2, 1]   # / 4
    })
    factor_graph, losses_from_training = models.factor.DiscreteFactorGraph.learn(
        (ab_dim, bc_dim, ca_dim),
        data
    )
    true_probs = (
        np.array([3.0 / 5, 2.0 / 5]),
        np.array([2.0 / 5, 2.0 / 5, 1.0 / 5]),
        np.array([1.0 / 5, 2.0 / 5, 1.0 / 5, 1.0 / 5])
    )

    for (variable, prob) in zip(data.columns, true_probs):
        models.visualization.probability_compare(
            factor_graph,
            variable,
            prob,
        )

    # demonstrate kl divergence calculations
    for pair, dim in zip(["ab", "bc", "ca"], [(2, 3), (3, 4), (4, 2)]):

        val_1, val_2 = pair
        d1, d2 = dim

        prob_pair = pd.crosstab(
            pd.Categorical(data[val_1].values, categories=list(range(d1))),
            pd.Categorical(data[val_2].values, categories=list(range(d2))),
            dropna=False
        )
        prob_pair = prob_pair.values / prob_pair.values.sum()
        models.visualization.probability_compare(
            factor_graph,
            pair,
            prob_pair
        )

        pred_pair = factor_graph.query(pair)
        entropy = pred_pair.entropy()
        logging.info(f"Computed {pair} entropy: {entropy} bits")
