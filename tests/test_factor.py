
import logging

import cerulean
import pandas as pd
import pyro
import pytest


@pytest.mark.factor
def test_factor_names():
    fs = "ab"
    dim = (2, 3)

    data = pd.DataFrame({
        "a": [1, 0], 
        "b": [1, 1]
    })

    factory = cerulean.dimensions.DimensionsFactory("a", "b")
    factory("a", dim[0])
    factory("b", dim[1])
    
    graph = cerulean.factor.DiscreteFactorGraph.learn(
        (factory(("a",)), factory(("b",)), factory(("a", "b"))),
        data,
        train_options={"num_iterations": 2}
    )

    # check pyro param store for correctly-named factors
    ps = pyro.get_param_store()
    parameters = set(ps.keys())
    assert parameters == {"f_a", "f_b", "f_ab"}
    logging.info(f"Currently in param store: {parameters}")

    pyro.clear_param_store()

    a = cerulean.dimensions.VariableDimensions("a", 3)
    b = cerulean.dimensions.VariableDimensions("b", 3)
    c = cerulean.dimensions.VariableDimensions("c", 3)

    for fname, prefix in zip(
        ["all_equal", "all_different"],
        ["ALL_EQUAL", "ALL_DIFFERENT"]
    ):
        some_factors = getattr(cerulean.constraint, fname)(a, b, c)
        some_ids = set(
            cerulean.factor.make_factor_name(factor.fs, factor._factor_name_prefix)
            for factor in some_factors
        )
        assert some_ids == {f"{prefix}_ab", f"{prefix}_bc", f"{prefix}_ac"}
        logging.info(f"Generated IDs: {some_ids}")

