"""Portable exact-inference checks independent of downloaded NBA data."""

from itertools import product

import numpy as np
import pandas as pd
import pytest

from src.variable_elimination import BayesianNetwork, PARENTS


@pytest.fixture
def synthetic_shots():
    states = {
        "T": ["2", "3"], "D": ["<5", "25-30"], "C": ["0-2", "6+"],
        "S": ["0-4", "22+"], "B": ["0", "7+"], "H": ["<2", "6+"],
    }
    rows = []
    for values in product(*states.values()):
        row = dict(zip(states, values))
        # Every state is supported, with nonuniform conditional probabilities.
        makes = 1 + 2 * (row["D"] == "<5") + (row["C"] == "6+")
        misses = 1 + (row["H"] == "6+") + (row["T"] == "3")
        rows.extend([{**row, "M": "1"}] * makes)
        rows.extend([{**row, "M": "0"}] * misses)
    result = pd.DataFrame(rows)
    result["GAME_ID"] = np.arange(len(result)) // 8
    return result


def brute_force_query(model, target, evidence=None):
    """Construct the entire joint, then condition and marginalize."""
    joint = pd.DataFrame({"_join": [1]})
    for node, factor in model.factors.items():
        table = factor.table.rename(columns={"p": f"p_{node}"})
        shared = [column for column in joint.columns if column in table.columns]
        joint = joint.merge(table, on=shared) if shared else joint.merge(table, how="cross")
    joint["joint_p"] = joint[[f"p_{node}" for node in model.factors]].prod(axis=1)
    for variable, value in (evidence or {}).items():
        joint = joint.loc[joint[variable].eq(str(value))]
    result = joint.groupby(target)["joint_p"].sum()
    return (result / result.sum()).to_dict()


@pytest.mark.parametrize("target,evidence", [
    ("M", {}), ("M", {"T": "3"}), ("M", {"D": "<5"}),
    ("M", {"T": "3", "D": "25-30", "H": "<2"}),
    ("M", {"D": "<5", "T": "2", "C": "0-2", "H": "<2"}),
    ("M", {"S": "0-4", "B": "7+"}),
    ("S", {"M": "1", "H": "6+"}), ("D", {"M": "0", "C": "6+"}),
])
def test_ve_matches_brute_force(synthetic_shots, target, evidence):
    model = BayesianNetwork(alpha=1.0).fit(synthetic_shots)
    actual = model.query(target, evidence)
    expected = brute_force_query(model, target, evidence)
    assert actual.keys() == expected.keys()
    np.testing.assert_allclose(list(actual.values()), [expected[s] for s in actual], atol=1e-12)
    assert sum(actual.values()) == pytest.approx(1.0)


def test_ve_matches_every_make_cpt_row(synthetic_shots):
    """With all make-node parents observed, ancestors integrate out."""
    model = BayesianNetwork(alpha=1.0).fit(synthetic_shots)
    cpt = model.factors["M"].table
    for _, row in cpt.loc[cpt["M"].eq("1")].iterrows():
        evidence = {parent: row[parent] for parent in PARENTS["M"]}
        assert model.query("M", evidence)["1"] == pytest.approx(row["p"], abs=1e-12)


@pytest.mark.parametrize("target,evidence,match", [
    ("M", {"M": "1"}, "Target cannot"),
    ("M", {"D": "invalid"}, "Unknown state"),
    ("M", {"unknown": "0"}, "Unknown variable"),
])
def test_ve_rejects_invalid_evidence(synthetic_shots, target, evidence, match):
    model = BayesianNetwork(alpha=1.0).fit(synthetic_shots)
    with pytest.raises(ValueError, match=match):
        model.query(target, evidence)
