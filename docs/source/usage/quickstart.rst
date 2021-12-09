Quickstart
==========
After `installation <Installation>`_, a user can run the unit tests.

.. code-block:: bash

    $ python -m pytest --cov=cerulean
    
An end-to-end example
=====================

NOTE: to build documentation that includes the image at the end, first
run the example `end-to-end-basic.py` in the `examples` directory!

* Step 0: get some data (here we'll just make some up). For our purposes, we'll 
  use a tiny little data frame that looks like::

                                    NYSE  NASDAQ    BATS
        2021-11-11 07:00:04.100000  100.10  100.10  100.09
        2021-11-11 07:00:04.100001  100.10  100.10  100.09
        2021-11-11 07:00:04.100002  100.10  100.10  100.09
        2021-11-11 07:00:04.100003  100.11  100.11  100.11
        2021-11-11 07:00:04.100004  100.11  100.11  100.11
        2021-11-11 07:00:04.100005  100.11  100.11  100.11
        2021-11-11 07:00:04.100006  100.11  100.11  100.11
        2021-11-11 07:00:04.100007  100.11  100.11  100.11
        2021-11-11 07:00:04.100008  100.11  100.11  100.11
        2021-11-11 07:00:04.100009  100.11  100.11  100.11
        2021-11-11 07:00:04.100010  100.11  100.11  100.11
        2021-11-11 07:00:04.100011  100.11  100.11  100.11
        2021-11-11 07:00:04.100012  100.10  100.11  100.10
        2021-11-11 07:00:04.100013  100.09  100.10  100.10
        2021-11-11 07:00:04.100014  100.09  100.10  100.10
        2021-11-11 07:00:04.100015  100.08  100.09  100.09

  after resampling at the microsecond level. 
* Step 1: make a stationarizing transform and its inverse, and compute the 
  stationarized version of the time series:

  .. code-block:: python

    s, s_inv = cerulean.transform.make_stationarizer("logdiff")
    stationary, centralized = cerulean.transform.to_stationary(
        np.mean, s, prices
    )

  ...resulting in a dataframe that looks like::

                                        NYSE   NASDAQ     BATS
        2021-11-11 07:00:04.100001  0.003330  0.00333 -0.00666
        2021-11-11 07:00:04.100002  0.003330  0.00333 -0.00666
        2021-11-11 07:00:04.100003  0.013320  0.01332  0.01332
        2021-11-11 07:00:04.100004  0.000000  0.00000  0.00000
        2021-11-11 07:00:04.100005  0.000000  0.00000  0.00000
        2021-11-11 07:00:04.100006  0.000000  0.00000  0.00000
        2021-11-11 07:00:04.100007  0.000000  0.00000  0.00000
        2021-11-11 07:00:04.100008  0.000000  0.00000  0.00000
        2021-11-11 07:00:04.100009  0.000000  0.00000  0.00000
        2021-11-11 07:00:04.100010  0.000000  0.00000  0.00000
        2021-11-11 07:00:04.100011  0.000000  0.00000  0.00000
        2021-11-11 07:00:04.100012 -0.009990  0.00000 -0.00999
        2021-11-11 07:00:04.100013 -0.013320 -0.00333 -0.00333
        2021-11-11 07:00:04.100014 -0.006660  0.00333  0.00333
        2021-11-11 07:00:04.100015 -0.016652 -0.00666 -0.00666

* Step 2: Now discretize the stationary data. You can either let `cerulean`
  figure out the values to use as min / max bin edges (not recommended) or you can
  do it yourself, as we do here. 

  .. code-block:: python 

    the_min = -0.02
    the_max = 0.02
    # you also need to choose how many bins to have.
    n_bins = 10
    n_cutpoints = n_bins + 1
    discrete_stationary = cerulean.transform.continuous_to_variable_level(
        stationary,
        n_cutpoints,
        the_max=the_max,
        the_min=the_min,
    )

* Step 3: build the factor graph. We will assume a fully-connected three-node
  graph with one factor for each pair of nodes. We specify structure using
  `Dimensions` objects, then learn its parameters from data.

  .. code-block:: python

    dim_factory = cerulean.dimensions.DimensionsFactory(*discrete_stationary.columns)
    # Register the variable's dimensionality with the dimension factory
    for location in discrete_stationary.columns:
        dim_factory(location, n_cutpoints)
    # create the appropriate factor dimensions
    factor_dims = [
        dim_factory(loc_pair)
        for loc_pair in itertools.combinations(discrete_stationary.columns, 2)
    ]
    # actually create the factor graph, learning its parameters from data
    variable_mapping = dim_factory.mapping()
    factor_graph, losses_from_training = cerulean.factor.DiscreteFactorGraph.learn(
        factor_dims,
        discrete_stationary.rename(columns=variable_mapping)
    )

  During learning you get some output telling you about how it's going::

    INFO:root:On iteration 0, -log p(x) = 215.81060028076172
    INFO:root:On iteration 100, -log p(x) = 117.58047103881836
    INFO:root:On iteration 200, -log p(x) = 73.17794418334961
    INFO:root:On iteration 300, -log p(x) = 67.42023849487305
    INFO:root:On iteration 400, -log p(x) = 66.187255859375
    INFO:root:On iteration 500, -log p(x) = 65.66991424560547
    INFO:root:On iteration 600, -log p(x) = 65.39154052734375
    INFO:root:On iteration 700, -log p(x) = 65.22031211853027
    INFO:root:On iteration 800, -log p(x) = 65.10572814941406
    INFO:root:On iteration 900, -log p(x) = 65.02443504333496

  Afterward you can check out the dimensionality and structure of the factor graph
  you specified. (This isn't very useful here, but it could be for a more complicated
  model.)::

    INFO:root:Learned a factor graph:
    DiscreteFactorGraph(id=DiscreteFactorGraph1
            DiscreteFactor(name=f_DiscreteFactor(ab), fs=ab, dim=(11, 11)),
            DiscreteFactor(name=f_DiscreteFactor(ac), fs=ac, dim=(11, 11)),
            DiscreteFactor(name=f_DiscreteFactor(bc), fs=bc, dim=(11, 11)),
    )

* Step 4: run inference! This depends entirely on the questions that you want to answer
  and is really domain-specific, so we won't cover all the things you could do here. 
  We will just give one illustrative example. First, we'll infer all marginal distributions
  before observing any new data. Remember, we're still working with the stationarized
  discrete time series here (moving back into nonstationary territory is the provenance
  of `PROPER` not `cerulean`).

  .. code-block:: python

    prior_predictive_marginals = {
        name : factor_graph.query(variable_mapping[name])
        for name in discrete_stationary.columns
    }
    prior_predictive = {
        name: f.table for (name, f) in prior_predictive_marginals.items()
    }

  Taking a look at the prior predictive distribution::

    INFO:root:Inferred prior predictive:
    {'NYSE': tensor([3.6398e-04, 6.6892e-02, 6.6474e-02, 6.6711e-02, 6.6893e-02, 5.3160e-01,
            1.3316e-01, 3.6398e-04, 3.6398e-04, 6.6820e-02, 3.6398e-04]), 'NASDAQ': tensor([4.0922e-04, 4.0922e-04, 4.0922e-04, 4.0922e-04, 6.6839e-02, 6.6394e-01,
            1.9955e-01, 4.0922e-04, 4.0922e-04, 6.6808e-02, 4.0922e-04]), 'BATS': tensor([3.7743e-04, 3.7743e-04, 3.7743e-04, 6.7051e-02, 1.9956e-01, 5.9748e-01,
            6.6841e-02, 3.7743e-04, 3.7743e-04, 6.6808e-02, 3.7743e-04])}

  Okay, fine. Now, suppose that we observe that `BATS = 4` (i.e., that after discretization,
  the BATS observation falls into category 4 out of 11). We first post this as
  evidence against a **new graph** created using `snapshot`, then run inference against
  this graph.

  .. code-block:: python
    
    new_graph = factor_graph.snapshot()
    new_graph.post_evidence(variable_mapping["BATS"], torch.tensor(4))
    posterior_predictive_marginals = {
        name : new_graph.query(variable_mapping[name])
        for name in ["NYSE", "NASDAQ"]
    }
    posterior_predictive = {
        name: f.table for (name, f) in posterior_predictive_marginals.items()
    }

  Let's see if the posterior predictive looks any different from the prior::

    INFO:root:Inferred posterior predictive:
    {'NYSE': tensor([2.1006e-04, 3.3232e-01, 1.2898e-04, 1.4073e-04, 1.6563e-03, 1.2151e-03,
            6.6354e-01, 2.1006e-04, 2.1006e-04, 1.6563e-04, 2.1006e-04]), 'NASDAQ': tensor([2.2888e-04, 2.2888e-04, 2.2888e-04, 2.2888e-04, 3.3188e-01, 1.4793e-03,
            6.6487e-01, 2.2888e-04, 2.2888e-04, 1.7453e-04, 2.2888e-04])}

  Indeed it does. To see this a little better, we can do what we love best and make
  a pretty picture:

  .. image:: ../../../examples/end_to_end_basic/end-to-end-basic-marginal.png