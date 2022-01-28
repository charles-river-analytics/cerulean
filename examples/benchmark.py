
import logging
import pathlib
import sys
import time

import cerulean
import numpy as np
import pandas as pd
import torch


logging.getLogger().setLevel(logging.INFO)
PATH = pathlib.Path("./benchmark_results")
PATH.mkdir(exist_ok=True, parents=True,)


def er_graph(n: int, d: int, p: float) -> cerulean.factor.FactorGraph:
    assert n > 1
    assert d > 1
    assert (p > 0) and (p < 1)
    factory = cerulean.dimensions.DimensionsFactory(
        *[f"var_{i}" for i in range(n)]
    )
    for ix in range(n):
        factory(f"var_{ix}", d)
    factors = []
    for i in range(n-1):
        for j in range(i + 1, n):
            if torch.rand((1,)) < p:
                factors.append(
                    cerulean.constraint.ConstraintFactor.random(
                        factory((f"var_{i}", f"var_{j}"))
                    )
                )
    graph = cerulean.factor.DiscreteFactorGraph(*factors, precompute_path=False, inference_cache_size=128,)
    vars = list(graph.factors.values())[0]._variables
    t0 = time.perf_counter()
    try:
        graph.build_contract_expr(result_spec=vars[0], optimize="random-greedy-128")
    except KeyboardInterrupt:
        t1 = time.perf_counter()
        logging.info(f"Took {60 * (t1 - t0)}m of compile time before quitting out of frustration.")
        sys.exit()
    t1 = time.perf_counter()
    logging.info(f"Took {t1 - t0}s to compile contraction path.")
    return graph


def to_ms(t0, t1):
    return round(1000 * (t1 - t0), 4)


def er_graph_numnodes_results():
    ns = list(range(10, 20 + 1))
    d = 10
    p = 0.3
    results = pd.DataFrame(
        columns=ns,
        index=["query", "similar query"]
    )

    for n in ns:
        logging.info(f"Num nodes: on n = {n}")
        first_res = []
        second_res = []

        for rerun in range(3):
            graph = er_graph(n, d, p)
            vars = list(graph.factors.values())[0]._variables
            logging.info(graph)

            t0 = time.perf_counter()
            _ = graph.query(vars[0], safe=False)
            t1 = time.perf_counter()
            first_res.append(to_ms(t0, t1))
            logging.info(f"Took {to_ms(t0, t1)}ms to run query first time.")

            t0 = time.perf_counter()
            _ = graph.query(vars[1], safe=False)
            t1 = time.perf_counter()
            second_res.append(to_ms(t0, t1))
            logging.info(f"Took {to_ms(t0, t1)}ms to run similar query.")

        results[n].loc["query"] = np.mean(first_res)
        results[n].loc["similar query"] = np.mean(second_res)

    results.to_csv(PATH / "num_nodes_scaling.csv")


def er_graph_dim_results():
    n = 10
    ds = [5, 10, 25, 50, 100]
    p = 0.3
    results = pd.DataFrame(
        columns=ds,
        index=["query", "similar query"]
    )

    for d in ds:
        logging.info(f"Dim: on d = {d}")
        first_res = []
        second_res = []

        for rerun in range(3):
            graph = er_graph(n, d, p)
            vars = list(graph.factors.values())[0]._variables
            logging.info(graph)

            t0 = time.perf_counter()
            _ = graph.query(vars[0], safe=False)
            t1 = time.perf_counter()
            first_res.append(to_ms(t0, t1))
            logging.info(f"Took {to_ms(t0, t1)}ms to run query first time.")

            t0 = time.perf_counter()
            _ = graph.query(vars[1], safe=False)
            t1 = time.perf_counter()
            second_res.append(to_ms(t0, t1))
            logging.info(f"Took {to_ms(t0, t1)}ms to run similar query.")

        results[d].loc["query"] = np.mean(first_res)
        results[d].loc["similar query"] = np.mean(second_res)

    results.to_csv(PATH / "dim_scaling.csv")


def er_graph_prob_results():
    n = 10
    d = 10
    ps = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    results = pd.DataFrame(
        columns=ps,
        index=["query", "similar query"]
    )

    for p in ps:
        logging.info(f"Prob: on p = {p}")
        first_res = []
        second_res = []

        for rerun in range(3):
            graph = er_graph(n, d, p)
            vars = list(graph.factors.values())[0]._variables
            logging.info(graph)

            t0 = time.perf_counter()
            _ = graph.query(vars[0], safe=False)
            t1 = time.perf_counter()
            first_res.append(to_ms(t0, t1))
            logging.info(f"Took {to_ms(t0, t1)}ms to run query first time.")

            t0 = time.perf_counter()
            _ = graph.query(vars[1], safe=False)
            t1 = time.perf_counter()
            second_res.append(to_ms(t0, t1))
            logging.info(f"Took {to_ms(t0, t1)}ms to run similar query.")

        results[p].loc["query"] = np.mean(first_res)
        results[p].loc["similar query"] = np.mean(second_res)

    results.to_csv(PATH / "prob_scaling.csv")


def main():
    # er_graph_numnodes_results()
    # er_graph_dim_results()
    er_graph_prob_results()


if __name__ == "__main__":
    main()