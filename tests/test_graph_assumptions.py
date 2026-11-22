"""Graph reasoning and numerical identities without the NBA data file."""

from itertools import product

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from src.graph_assumptions import build_claims, graph_for_model, verify_claims
from src.graphs import FEATURES, HAND_EDGES, HAND_PARENT_MAP, HAND_PARENTS, VARIABLES
from src.naive_bayes import NaiveBayesShotModel
from src.structure_learning import ConditionalMakeModel, prepare_data
from src.variable_elimination import BayesianNetwork


@pytest.mark.parametrize("edges,given,expected", [
    ([("A", "B"), ("B", "C")], set(), False),
    ([("A", "B"), ("B", "C")], {"B"}, True),
    ([("B", "A"), ("B", "C")], set(), False),
    ([("B", "A"), ("B", "C")], {"B"}, True),
    ([("A", "B"), ("C", "B")], set(), True),
    ([("A", "B"), ("C", "B")], {"B"}, False),
    ([("A", "B"), ("C", "B"), ("B", "D")], {"D"}, False),
])
def test_d_separation_motifs(edges, given, expected):
    assert nx.is_d_separator(nx.DiGraph(edges), {"A"}, {"C"}, given) == expected


def test_hand_graph_has_one_authoritative_factorization():
    expected = {(parent, node) for node, parents in HAND_PARENT_MAP.items() for parent in parents}
    assert set(HAND_EDGES) == expected
    assert HAND_PARENT_MAP["M"] == HAND_PARENTS
    assert set(graph_for_model("Hand BN", [])) == set(VARIABLES)
    assert nx.is_directed_acyclic_graph(graph_for_model("Hand BN", []))


def test_all_project_claims_are_implied_by_the_actual_graphs():
    learned_parents = ["D", "T", "C", "S"]
    learned_edges = [("D", "T"), ("B", "H"), *[(node, "M") for node in learned_parents]]
    rows = verify_claims(learned_parents, learned_edges)
    assert len(rows) == 8
    assert all(row["d_separated"] for row in rows)
    assert all("not evidence" in row["interpretation"] for row in rows)
    joint = next(row for row in rows if row["id"] == "learned_joint")
    assert set(joint["y"].split(",")) == {"B", "H"}
    assert set(joint["given"].split(",")) == set(learned_parents)


def test_learned_claims_follow_parents_instead_of_hardcoded_touch_assumption():
    claims = build_claims(["H", "D"])
    assert "learned_touch" not in {claim["id"] for claim in claims}
    joint = next(claim for claim in claims if claim["id"] == "learned_joint")
    assert joint["given"] == ["H", "D"]
    assert set(joint["y"]) == set(FEATURES) - {"H", "D"}
    assert not any(claim["model"] == "Learned BN" for claim in build_claims(FEATURES))


@pytest.mark.parametrize("parents", [["M"], ["D", "D"], ["invalid"]])
def test_bad_parent_names_are_rejected(parents):
    with pytest.raises(ValueError, match="parents"):
        build_claims(parents)


def test_bad_graphs_and_parent_mismatch_are_rejected():
    with pytest.raises(ValueError, match="Unknown"):
        graph_for_model("Invalid model", [])
    for edges in [[("D", "T"), ("T", "D")], [("extra", "D")], [("M", "D")]]:
        with pytest.raises(ValueError):
            graph_for_model("Learned BN", edges)
    with pytest.raises(ValueError, match="does not match"):
        verify_claims(["D"], [("T", "M")])


def synthetic_data():
    """Full binary support plus unequal counts produce nontrivial fitted CPTs."""
    states = pd.DataFrame(product(["0", "1"], repeat=len(VARIABLES)), columns=VARIABLES)
    rng = np.random.default_rng(250)
    return states.loc[states.index.repeat(rng.integers(1, 9, len(states)))].reset_index(drop=True)


def fitted_hand_joint(model):
    joint = pd.DataFrame({"p": [1.0]})
    for factor in model.factors.values():
        shared = [column for column in factor.variables if column in joint]
        merged = joint.merge(factor.table, on=shared) if shared else joint.merge(factor.table, how="cross")
        merged["p"] = merged.pop("p_x") * merged.pop("p_y")
        joint = merged
    assert np.isclose(joint["p"].sum(), 1)
    return joint


def assert_numerical_independence(joint, x, y, given):
    """Compare joint conditional probabilities with products of marginals."""
    joint = joint.copy()
    if not given:
        joint["_condition"] = "all"
        given = ["_condition"]
    pxy = joint.groupby(given + x + y, as_index=False)["p"].sum()
    pz = joint.groupby(given, as_index=False)["p"].sum().rename(columns={"p": "pz"})
    pxz = joint.groupby(given + x, as_index=False)["p"].sum().rename(columns={"p": "pxz"})
    pyz = joint.groupby(given + y, as_index=False)["p"].sum().rename(columns={"p": "pyz"})
    combined = pxy.merge(pz, on=given).merge(pxz, on=given + x).merge(pyz, on=given + y)
    np.testing.assert_allclose(combined["p"] * combined["pz"], combined["pxz"] * combined["pyz"], atol=1e-12, rtol=1e-12)


def test_fitted_hand_bn_joint_satisfies_all_hand_claims():
    model = BayesianNetwork().fit(synthetic_data())
    joint = fitted_hand_joint(model)
    for claim in build_claims([]):
        if claim["model"] == "Hand BN":
            assert_numerical_independence(joint, claim["x"], claim["y"], claim["given"])
    evidence = {"D": "0", "T": "1", "C": "0", "H": "1"}
    reference = model.query("M", evidence)
    for clock, dribbles in product(["0", "1"], repeat=2):
        extra = model.query("M", {**evidence, "S": clock, "B": dribbles})
        np.testing.assert_allclose(list(extra.values()), list(reference.values()), atol=1e-12)


def test_fitted_naive_bayes_joint_satisfies_feature_independence_given_make():
    data = synthetic_data()
    model = NaiveBayesShotModel().fit(data[FEATURES], data["M"].astype(int))
    rows = []
    # Include the reserved unseen category so the fitted generative joint sums to 1.
    for values in product(["0", "1", "unseen"], repeat=len(FEATURES)):
        for outcome in [0, 1]:
            probability = np.exp(model.log_prior[outcome])
            for feature, value in zip(FEATURES, values):
                probability *= np.exp(model.log_cpts[feature][outcome].get(value, model.unknown_log_probs[feature][outcome]))
            rows.append({**dict(zip(FEATURES, values)), "M": str(outcome), "p": probability})
    joint = pd.DataFrame(rows)
    assert np.isclose(joint["p"].sum(), 1)
    for claim in build_claims([]):
        if claim["model"] == "Naive Bayes":
            assert_numerical_independence(joint, claim["x"], claim["y"], claim["given"])


def test_fitted_learned_make_cpt_is_invariant_to_omitted_features():
    data = prepare_data(synthetic_data())
    parents = ["D", "T", "S", "C"]
    model = ConditionalMakeModel(parents).fit(data)
    baseline = model.predict_proba(data)
    changed = data.copy()
    changed["H"] = changed["H"].map({"0": "1", "1": "0"})
    changed["B"] = changed["B"].map({"0": "1", "1": "0"})
    np.testing.assert_allclose(baseline, model.predict_proba(changed), atol=1e-12)
