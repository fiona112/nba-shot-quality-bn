
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluate import evaluate_model


DATA_DIR = Path("data/processed")
RESULTS_DIR = Path("results/tables")

# Parents of M in our hand-designed Bayesian network
MAKE_PARENTS = ["D", "T", "C", "H"]

ALPHA = 1.0


class HandDesignedBN:
    """
    First implementation of the shot-outcome CPT:

        P(M | D, T, C, H)

    With all four parents observed, this CPT is enough
    to predict the make probability under our proposed BN.

    Full-graph inference will be implemented separately.
    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.cpt = None
        self.training_rate = None

    def fit(self, train):
        self.training_rate = train["M"].mean()

        # Count made shots and total attempts for
        # each combination of parent variables.
        counts = (
            train.groupby(
                MAKE_PARENTS, observed=True
            )["M"]
            .agg(["sum", "count"])
            .reset_index()
        )

        # Laplace-smoothed P(M=1 | parents)
        counts["p_make"] = (
            (counts["sum"] + self.alpha)
            / (counts["count"] + 2 * self.alpha)
        )

        self.cpt = counts
        return self

    def predict_proba(self, data):
        if self.cpt is None:
            raise RuntimeError("Fit the model first.")

        # Look up the CPT entry for each shot.
        merged = data[MAKE_PARENTS].merge(
            self.cpt,
            on=MAKE_PARENTS,
            how="left",
            sort=False
        )

        # Unseen parent combinations have no training
        # observations. Their Laplace-smoothed estimate
        # under a symmetric prior is 0.5.
        probabilities = merged["p_make"].fillna(0.5)

        return probabilities.to_numpy(dtype=float)


def main():
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")

    model = HandDesignedBN(alpha=ALPHA)
    model.fit(train)

    probabilities = model.predict_proba(test)

    metrics = evaluate_model(
        "Hand-Designed BN (Make CPT)",
        test["M"],
        probabilities
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([{
        "model": "Hand-Designed BN",
        **metrics
    }]).to_csv(
        RESULTS_DIR / "hand_bn_metrics.csv",
        index=False
    )

    model.cpt.to_csv(
        RESULTS_DIR / "make_cpt.csv",
        index=False
    )

    print("\n=== CPT DIAGNOSTICS ===")
    print("Observed parent combinations:", len(model.cpt))
    print("Combinations with fewer than 10 shots:",
          (model.cpt["count"] < 10).sum())
    print("Combinations with fewer than 30 shots:",
          (model.cpt["count"] < 30).sum())

    # 6 distance bins × 2 shot types ×
    # 4 defender bins × 3 touch-time bins
    possible_combinations = 6 * 2 * 4 * 3

    print("Possible combinations:", possible_combinations)
    print("Unobserved combinations:",
          possible_combinations - len(model.cpt))

    print("\nExample CPT entries:")
    print(
        model.cpt.sort_values(
            "count", ascending=False
        ).head(10).to_string(index=False)
    )

    print("\nResults saved successfully!")


if __name__ == "__main__":
    main()
