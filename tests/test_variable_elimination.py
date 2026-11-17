
import numpy as np
import pandas as pd

from src.variable_elimination import (
    BayesianNetwork,
    PARENTS
)


def brute_force_query(model, target, evidence=None):
    """
    Independently compute P(target | evidence)
    by constructing the full joint distribution
    and marginalizing unwanted variables.
    """
    evidence = evidence or {}

    joint = pd.DataFrame({"_join": [1]})

    for node, factor in model.factors.items():
        table = factor.table.copy()
        table = table.rename(columns={"p": f"p_{node}"})

        shared = [
            col for col in joint.columns
            if col in table.columns
        ]

        # Initial dummy column only
        if "_join" in shared:
            shared.remove("_join")

        if shared:
            joint = joint.merge(table, on=shared)
        else:
            joint = joint.merge(table, how="cross")

    probability_cols = [
        f"p_{node}" for node in PARENTS
    ]

    joint["joint_p"] = joint[probability_cols].prod(axis=1)

    for variable, value in evidence.items():
        joint = joint[
            joint[variable] == str(value)
        ]

    result = joint.groupby(target)["joint_p"].sum()
    result = result / result.sum()

    return result.to_dict()


def test_ve_matches_brute_force():
    train = pd.read_csv("data/processed/train.csv")

    model = BayesianNetwork(alpha=1.0)
    model.fit(train)

    queries = [
        {},
        {"T": "3"},
        {"D": "<5"},
        {"T": "3", "D": "30+"},
        {"T": "3", "D": "30+", "H": "<2"},
        {"D": "<5", "T": "2", "C": "2-4", "H": "<2"},
        {"S": "0-4", "B": "7+"},
        {"S": "22+", "T": "2", "C": "0-2"},
    ]

    for i, evidence in enumerate(queries, start=1):
        ve = model.query("M", evidence)
        brute = brute_force_query(model, "M", evidence)

        for state in ["0", "1"]:
            assert np.isclose(
                ve[state],
                brute[state],
                atol=1e-10
            ), (
                f"Query {i} failed for state {state}: "
                f"VE={ve[state]}, brute={brute[state]}"
            )

        print(f"Query {i}: PASS")

    print("\nAll VE vs. brute-force checks passed!")


def test_ve_matches_make_cpt():
    train = pd.read_csv("data/processed/train.csv")

    model = BayesianNetwork(alpha=1.0)
    model.fit(train)

    make_cpt = model.factors["M"].table

    # Check several CPT entries directly
    for _, row in make_cpt.head(20).iterrows():
        if row["M"] != "1":
            continue

        evidence = {
            p: str(row[p]) for p in PARENTS["M"]
        }

        predicted = model.query("M", evidence)["1"]

        assert np.isclose(
            predicted, row["p"], atol=1e-10
        )

    print("VE matches the make CPT!")


if __name__ == "__main__":
    test_ve_matches_brute_force()
    test_ve_matches_make_cpt()
