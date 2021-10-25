import collections
import logging

import pytest
import torch

import models

def test_integration_1():
    da, db, dc = 2, 3, 4
    fs2dim = collections.OrderedDict({
        "ab": (da, db),  # shape = 2 * 3 = 6
        "bc": (db, dc),  # shape = 3 * 4 = 12
        "ca": (dc, da)   # shape = 4 * 2 = 8
    })
    data = collections.OrderedDict({
        "ab": torch.tensor([1, 1, 2, 1, 5]),
        "bc": torch.tensor([5, 5, 6, 4, 2]),
        "ca": torch.tensor([6, 6, 7, 2, 2])
    })
    factor_graph, losses_from_training = models.factor.FactorGraph.learn(
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