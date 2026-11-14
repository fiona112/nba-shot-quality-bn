"""Compare distance buckets on game-level validation splits of training data."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.bayesian_network import HandDesignedBN
from src.naive_bayes import NaiveBayesShotModel
from src.preprocess import FEATURES, SHOT_DISTANCE_BINS, SHOT_DISTANCE_LABELS


SCHEMES = {
    "original": (
        [0, 4, 10, 16, 22, 26, np.inf],
        ["<4", "4-10", "10-16", "16-22", "22-26", "26+"],
    ),
    "five_foot_25plus": (
        [0, 5, 10, 15, 20, 25, np.inf],
        ["<5", "5-10", "10-15", "15-20", "20-25", "25+"],
    ),
    "five_foot_30plus": (SHOT_DISTANCE_BINS, SHOT_DISTANCE_LABELS),
    "training_quantiles_7": None,
}
# Seed 123 reproduces the project's existing train_fit/validation split.
SEEDS = [123, 7, 42, 2026, 31415]
OUTPUT_PATH = Path("results/tables/distance_binning_validation.csv")


def compare_split(train, validation, schemes=None):
    results = []
    if schemes is None:
        schemes = SCHEMES
    for scheme, definition in schemes.items():
        fit = train.copy()
        val = validation.copy()
        if definition is None:
            # Learn cutoffs from this fold's training shots only.
            _, bins = pd.qcut(fit["SHOT_DIST"], q=7, retbins=True, duplicates="drop")
            bins[0], bins[-1] = 0, np.inf
            labels = [f"q{i + 1}" for i in range(len(bins) - 1)]
        else:
            bins, labels = definition
        for frame in (fit, val):
            frame["D"] = pd.cut(
                frame["SHOT_DIST"], bins=bins, labels=labels, right=False
            ).astype("string")

        x_fit = fit[FEATURES].fillna("missing").astype(str)
        x_val = val[FEATURES].fillna("missing").astype(str)
        logistic = Pipeline([
            ("encoder", ColumnTransformer([
                ("categories", OneHotEncoder(handle_unknown="ignore"), FEATURES)
            ])),
            ("model", LogisticRegression(max_iter=1000)),
        ])
        logistic.fit(x_fit, fit["M"])
        naive = NaiveBayesShotModel().fit(x_fit, fit["M"])
        hand = HandDesignedBN().fit(fit)
        predictions = {
            "Hand BN": hand.predict_proba(val),
            "Logistic regression": logistic.predict_proba(x_val)[:, 1],
            "Naive Bayes": naive.predict_proba(x_val)[:, 1],
        }
        for model, probabilities in predictions.items():
            probabilities = np.clip(probabilities, 1e-10, 1 - 1e-10)
            results.append({
                "scheme": scheme,
                "model": model,
                "distance_cutoffs": ",".join(str(edge) for edge in bins),
                "training_shots": len(fit),
                "validation_shots": len(val),
                "clock_preprocessing": "period_proxy_else_drop_missing_keep_flagged_24",
                "log_loss": log_loss(val["M"], probabilities, labels=[0, 1]),
                "brier_score": brier_score_loss(val["M"], probabilities),
                "accuracy": accuracy_score(val["M"], probabilities >= 0.5),
            })
    return results


def main():
    # Keep the held-out test set out of bucket selection entirely.
    data = pd.read_csv("data/processed/train.csv")
    results = []
    for seed in SEEDS:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
        fit_idx, validation_idx = next(splitter.split(data, groups=data["GAME_ID"]))
        train = data.iloc[fit_idx]
        validation = data.iloc[validation_idx]
        assert set(train["GAME_ID"]).isdisjoint(validation["GAME_ID"])
        results.extend(
            {"seed": seed, **row} for row in compare_split(train, validation)
        )
        print(f"Completed validation seed {seed}", flush=True)

    results = pd.DataFrame(results)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_PATH, index=False)
    print("\nExisting validation split (seed 123):")
    print(results.loc[results["seed"].eq(123)].to_string(index=False))
    print("\nMean metrics across five overlapping game-level validation splits:")
    print(results.groupby(["model", "scheme"])[
        ["log_loss", "brier_score", "accuracy"]
    ].mean().to_string())
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
