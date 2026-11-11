
from pathlib import Path

import numpy as np
import pandas as pd

from pgmpy.estimators import (
    HillClimbSearch,
    ExpertKnowledge,
)

from src.evaluate import evaluate_model


DATA_DIR = Path("data/processed")
RESULTS_DIR = Path("results/tables")

FEATURES = ["T", "D", "C", "S", "B", "H"]
VARIABLES = FEATURES + ["M"]

HAND_PARENTS = ["D", "T", "C", "H"]

HAND_EDGES = [
    ("S", "B"),
    ("B", "H"),
    ("S", "H"),
    ("D", "T"),
    ("D", "C"),
    ("B", "C"),
    ("D", "M"),
    ("T", "M"),
    ("C", "M"),
    ("H", "M"),
]

ALPHA = 1.0


def prepare_data(df):
    """
    Convert every variable to a categorical string.

    For this structure-learning experiment, missing
    values are represented by the state "missing".
    """
    data = df[VARIABLES].copy()

    for col in FEATURES:
        data[col] = (
            data[col]
            .astype("string")
            .fillna("missing")
            .astype(str)
        )

    data["M"] = data["M"].astype(int).astype(str)

    return data


class ConditionalMakeModel:
    """
    Estimate P(M=1 | parents) using Laplace smoothing.

    This evaluates the make-node CPT of a given graph
    when its parent variables are observed.
    """

    def __init__(self, parents, alpha=1.0):
        self.parents = list(parents)
        self.alpha = alpha
        self.lookup = None
        self.prior = None
        self.states = {}

    def fit(self, data):
        self.prior = (
            (data["M"].eq("1").sum() + self.alpha)
            / (len(data) + 2 * self.alpha)
        )

        if not self.parents:
            return self

        self.states = {
            p: sorted(data[p].unique())
            for p in self.parents
        }

        counts = (
            data.assign(
                made=data["M"].eq("1").astype(int)
            )
            .groupby(self.parents, observed=True)["made"]
            .agg(["sum", "count"])
            .reset_index()
        )

        counts["p_make"] = (
            (counts["sum"] + self.alpha)
            / (counts["count"] + 2 * self.alpha)
        )

        self.lookup = counts[
            self.parents + ["p_make", "count"]
        ].copy()

        return self

    def predict_proba(self, data):
        if not self.parents:
            return np.full(len(data), self.prior)

        merged = data[self.parents].merge(
            self.lookup,
            on=self.parents,
            how="left",
            sort=False,
            validate="many_to_one",
        )

        # Unseen combinations get the symmetric
        # Laplace-smoothed binary probability, 0.5.
        return (
            merged["p_make"]
            .fillna(0.5)
            .to_numpy(dtype=float)
        )

    def coverage(self, data):
        if not self.parents:
            return 0.0

        matched = data[self.parents].merge(
            self.lookup,
            on=self.parents,
            how="left",
            sort=False,
            validate="many_to_one",
        )

        return float(matched["count"].isna().mean())


def learn_structure(train):
    # Do not let the outcome M point to shot features.
    forbidden = [
        ("M", feature)
        for feature in FEATURES
    ]

    expert = ExpertKnowledge(
        forbidden_edges=forbidden
    )

    print("\n=== LEARNING BN STRUCTURE ===")
    print("Method: Hill climbing")
    print("Score: BIC (discrete)")
    print("Maximum parents per node: 4")
    print("Outgoing edges from M: forbidden")

    search = HillClimbSearch(train)

    dag = search.estimate(
        scoring_method="bic-d",
        max_indegree=4,
        expert_knowledge=expert,
        show_progress=True,
    )

    edges = sorted(
        (str(u), str(v))
        for u, v in dag.edges()
    )

    parents_m = sorted(
        str(node) for node in dag.predecessors("M")
    )

    return dag, edges, parents_m


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    train_raw = pd.read_csv(
        DATA_DIR / "train_fit.csv"
    )
    val_raw = pd.read_csv(
        DATA_DIR / "validation.csv"
    )

    train = prepare_data(train_raw)
    val = prepare_data(val_raw)

    print("Training shots:", len(train))
    print("Validation shots:", len(val))

    print("\n=== ORIGINAL HAND-DESIGNED EDGES ===")
    for edge in sorted(HAND_EDGES):
        print(f"{edge[0]} -> {edge[1]}")

    dag, learned_edges, learned_parents = (
        learn_structure(train)
    )

    print("\n=== LEARNED EDGES ===")
    for u, v in learned_edges:
        print(f"{u} -> {v}")

    print("\n=== PARENTS OF MAKE (M) ===")
    print("Original:", HAND_PARENTS)
    print("Learned:", learned_parents)

    original_set = set(HAND_EDGES)
    learned_set = set(learned_edges)

    print("\n=== STRUCTURE COMPARISON ===")
    print("Edges in both:")
    for edge in sorted(original_set & learned_set):
        print(" ", edge)

    print("Only in original:")
    for edge in sorted(original_set - learned_set):
        print(" ", edge)

    print("Only in learned:")
    for edge in sorted(learned_set - original_set):
        print(" ", edge)

    results = []

    configurations = {
        "Hand BN (validation)": HAND_PARENTS,
        "Learned BN (validation)": learned_parents,
    }

    for name, parents in configurations.items():
        model = ConditionalMakeModel(
            parents=parents,
            alpha=ALPHA
        )
        model.fit(train)

        probabilities = model.predict_proba(val)
        metrics = evaluate_model(
            name,
            val["M"].astype(int),
            probabilities
        )

        unseen_fraction = model.coverage(val)

        print(
            f"Unseen validation parent combinations: "
            f"{100 * unseen_fraction:.3f}%"
        )

        results.append({
            "model": name,
            "parents": ",".join(parents),
            "unseen_fraction": unseen_fraction,
            **metrics,
        })

    pd.DataFrame(results).to_csv(
        RESULTS_DIR / "structure_validation.csv",
        index=False
    )

    pd.DataFrame(
        learned_edges,
        columns=["source", "target"]
    ).to_csv(
        RESULTS_DIR / "learned_edges.csv",
        index=False
    )

    print("\nResults saved:")
    print(
        RESULTS_DIR / "structure_validation.csv"
    )
    print(
        RESULTS_DIR / "learned_edges.csv"
    )


if __name__ == "__main__":
    main()
