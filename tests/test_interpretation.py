import json

import networkx as nx
import numpy as np
import pandas as pd
import pytest
from scipy.stats import beta

from src import interpretation
from src.variable_elimination import BayesianNetwork
from tests.test_variable_elimination import synthetic_shots


def test_cpt_support_probabilities_and_posterior(synthetic_shots):
    model = BayesianNetwork(alpha=1).fit(synthetic_shots)
    summary = interpretation.summarize_make_cpt(synthetic_shots, model)
    assert summary["training_shots"].sum() == len(synthetic_shots)
    assert (summary["training_shots"] > 0).all()
    row = summary.iloc[0]
    subset = synthetic_shots
    for parent in interpretation.HAND_PARENT_MAP["M"]:
        subset = subset.loc[subset[parent].eq(row[parent])]
    successes = subset["M"].eq("1").sum()
    failures = len(subset) - successes
    assert row["made"] == successes
    assert row["missed"] == failures
    assert row["laplace_make_probability"] == pytest.approx((successes + 1) / (len(subset) + 2))
    np.testing.assert_allclose(
        [row["posterior_ci95_lower"], row["posterior_ci95_upper"]],
        beta.ppf([0.025, 0.975], successes + 1, failures + 1),
    )


def test_queries_report_actual_supported_rows_and_full_parent_cpt(synthetic_shots):
    model = BayesianNetwork(alpha=1).fit(synthetic_shots)
    evidence = {"D": "<5", "T": "2", "C": "6+", "H": "<2"}
    queries, timings = interpretation.run_inference_queries(
        synthetic_shots, model, [("all", {}), ("full_parent", evidence)],
    )
    assert queries.iloc[0]["training_shots"] == len(synthetic_shots)
    assert queries.iloc[0]["training_games"] == synthetic_shots["GAME_ID"].nunique()
    row = queries.iloc[1]
    subset = synthetic_shots
    for variable, value in evidence.items():
        subset = subset.loc[subset[variable].eq(value)]
    assert row["training_shots"] == len(subset)
    assert row["empirical_make_rate"] == pytest.approx(subset["M"].eq("1").mean())
    assert row["ve_make_probability"] == pytest.approx((subset["M"].eq("1").sum() + 1) / (len(subset) + 2))
    assert json.loads(row["evidence"]) == evidence
    assert (timings["seconds"] >= 0).all()


def test_queries_reject_unsupported_examples(synthetic_shots):
    model = BayesianNetwork(alpha=1).fit(synthetic_shots)
    with pytest.raises(ValueError, match="no training support"):
        interpretation.run_inference_queries(synthetic_shots, model, [("invalid", {"D": "missing"})])


def test_subsamples_preserve_complete_games_and_are_reproducible(synthetic_shots):
    first = interpretation.sample_training_games(synthetic_shots, 13)
    second = interpretation.sample_training_games(synthetic_shots, 13)
    pd.testing.assert_frame_equal(first, second)
    assert 0 < first["GAME_ID"].nunique() < synthetic_shots["GAME_ID"].nunique()
    expected = synthetic_shots.loc[synthetic_shots["GAME_ID"].isin(first["GAME_ID"])].reset_index(drop=True)
    pd.testing.assert_frame_equal(first, expected)


def test_stability_uses_actual_returned_make_parents(synthetic_shots, monkeypatch):
    received = []
    def learner(data):
        received.append(data.copy())
        # Deliberately differs from the known full-training result.
        return nx.DiGraph([("H", "M")]), [("H", "M")], ["H"]
    monkeypatch.setattr(interpretation, "learn_structure", learner)
    edges, runs = interpretation.structure_stability(
        synthetic_shots, [("D", "M")], seeds=(13, 37), fraction=0.7,
    )
    assert len(received) == 2
    assert all(set(frame.columns) == set(interpretation.VARIABLES) for frame in received)
    assert runs["make_parents"].eq("H").all()
    indexed = edges.set_index(["source", "target"])
    assert indexed.loc[("H", "M"), "frequency"] == 1.0
    assert indexed.loc[("D", "M"), "frequency"] == 0.0
    assert bool(indexed.loc[("D", "M"), "full_train_fit_edge"])


def test_graph_figure_uses_every_actual_directed_edge(tmp_path, monkeypatch):
    """Check semantic content at the plotting boundary, not image pixels."""
    original = nx.draw_networkx
    drawn = []
    def capture(graph, **kwargs):
        drawn.append((set(graph.nodes), set(graph.edges)))
        return original(graph, **kwargs)
    monkeypatch.setattr(nx, "draw_networkx", capture)
    learned = [("H", "M"), ("T", "D")]
    path = tmp_path / "graphs.png"
    interpretation.plot_graphs(learned, path)
    assert path.stat().st_size > 0
    assert drawn == [
        (set(interpretation.VARIABLES), set(interpretation.HAND_EDGES)),
        (set(interpretation.VARIABLES), set(learned)),
    ]
