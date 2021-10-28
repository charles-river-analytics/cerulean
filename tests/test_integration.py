import collections
import logging
import time

import pytest
import torch

import models


def dims():
    return (2, 3, 4)


def get_fs2dim(dims):
    da, db, dc = dims
    return collections.OrderedDict({
        "ab": (da, db),  # shape = 2 * 3 = 6
        "bc": (db, dc),  # shape = 3 * 4 = 12
        "ca": (dc, da)   # shape = 4 * 2 = 8
    })


def get_data():
    return collections.OrderedDict({
        "ab": torch.tensor([1, 1, 2, 1, 5]),
        "bc": torch.tensor([5, 5, 6, 4, 2]),
        "ca": torch.tensor([6, 6, 7, 2, 2])
    })


def to_micro(t0, t1):
    return round(1e6 * (t1 - t0), 3)


def test_integration_1():
    fs2dim = get_fs2dim(dims())
    data = get_data()
    factor_graph, losses_from_training = models.factor.DiscreteFactorGraph.learn(
        fs2dim, data
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
    assert not torch.equal(normalized_clique_factor, p_clique)  # other marginal effects matter!

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