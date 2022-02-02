
import logging

import cerulean
import numpy as np
import pandas as pd
import pyro
import pytest
import torch


@pytest.mark.factor
@pytest.mark.slow
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


@pytest.mark.factor
@pytest.mark.slow
def test_column_rename_in_learn():
    dim = (2, 3)
    data = pd.DataFrame({
        "Q": [1, 0], 
        "X": [1, 1]
    })

    factory = cerulean.dimensions.DimensionsFactory("Q", "X")
    factory("Q", dim[0])
    factory("X", dim[1])
    
    with pytest.raises(KeyError):  # haven't mapped variables to dimension strings a, b, ab
        graph = cerulean.factor.DiscreteFactorGraph.learn(
            (factory(("Q",)), factory(("X",)), factory(("Q", "X"))),
            data,
            train_options={"num_iterations": 2}
        )

    graph_1, _ = cerulean.factor.DiscreteFactorGraph.learn(
        (factory(("Q",)), factory(("X",)), factory(("Q", "X"))),
        data.rename(columns=factory.mapping()),
        train_options={"num_iterations": 2}
    )
    logging.info(f"Learned graph after manually renaming data:\n{graph_1}")
    # should be equivalent to above
    graph_2, _ = cerulean.factor.DiscreteFactorGraph.learn(
        (factory(("Q",)), factory(("X",)), factory(("Q", "X"))),
        data,
        train_options={"num_iterations": 2},
        column_mapping=factory.mapping(),
    )
    logging.info(f"Learned graph after passing mapping:\n{graph_2}")


@pytest.mark.factor
@pytest.mark.slow
def test_link():
    # first graph
    data = pd.DataFrame({
        "a": [1, 0], 
        "b": [1, 1]
    })
    dim = (2, 3)

    factory = cerulean.dimensions.DimensionsFactory("a", "b")
    factory("a", dim[0])
    factory("b", dim[1])
    
    graph_1, _ = cerulean.factor.DiscreteFactorGraph.learn(
        (factory(("a",)), factory(("b",)), factory(("a", "b"))),
        data,
        train_options={"num_iterations": 2}
    )
    logging.info(f"Learned graph 1 from data:\n{graph_1}")

    # second graph
    constraint_factors = cerulean.constraint.all_different(
        factory.get_variable("a"), factory.get_variable("b")
    )
    graph_2 = cerulean.factor.DiscreteFactorGraph(*constraint_factors)
    logging.info(f"graph 2 is a constraint graph:\n{graph_2}")

    linked_graph = graph_1.link(graph_2)
    logging.info(f"Linked graph:\n{linked_graph}")
    joint_constraint = linked_graph.query("ab")
    logging.info(f"Joint distribution table *with* constraint:\n{joint_constraint.table}")
    assert torch.equal(joint_constraint.table[0, 0], torch.tensor(0.0))
    assert torch.equal(joint_constraint.table[1, 1], torch.tensor(0.0))

    # compare with non-linked graph
    joint = graph_1.query("ab")
    logging.info(f"Joint distribution table *without* constraint:\n{joint.table}")


class MockDataGenerator:

    def __init__(
        self,
        variables,
        batch_size=10, 
        num_batches=5,
        n_cutpoints=10,
        the_min=-3.0,
        the_max=3.0,
    ):
        self.variables = variables
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.n_cutpoints = n_cutpoints
        self.the_min = the_min
        self.the_max = the_max
    
    def __iter__(self,):
        for n in range(self.num_batches):
            the_dataframe = pd.DataFrame({
                v: np.random.randn(self.batch_size)
                for v in self.variables
            })
            yield cerulean.transform.continuous_to_variable_level(
                the_dataframe,
                self.n_cutpoints,
                the_min=self.the_min,
                the_max=self.the_max,
            )


def get_data_generator(
    variables,
    batch_size,
    num_batches,
    n_cutpoints=10,
    the_min=-3.0,
    the_max=3.0,
):
    return lambda id=0: MockDataGenerator(
        variables,
        batch_size=batch_size,
        num_batches=num_batches,
        n_cutpoints=n_cutpoints,
        the_min=the_min,
        the_max=the_max,
    )


@pytest.mark.train
@pytest.mark.slow
@pytest.mark.factor
def test_train_graph_from_generator():
    variables = ["X", "Y"]
    batch_size = 10
    num_batches = 50
    n_cutpoints = 11

    gen = get_data_generator(
        variables,
        batch_size,
        num_batches,
        n_cutpoints=n_cutpoints - 1,
    )
    
    factory = cerulean.dimensions.DimensionsFactory("X", "Y")
    factory("X", n_cutpoints)
    factory("Y", n_cutpoints)

    graph = cerulean.factor.DiscreteFactorGraph.learn(
        (factory(("X",)), factory(("Y",)), factory(("X", "Y"))),
        gen,
        train_options=dict(num_epochs=10, verbosity=100,),
        column_mapping=factory.mapping(),
    )
