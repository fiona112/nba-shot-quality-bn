
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.naive_bayes import NaiveBayesShotModel
from src.structure_learning import (
    ConditionalMakeModel,
    prepare_data,
)
from src.evaluate import evaluate_model


DATA_DIR = Path("data/processed")
FIG_DIR = Path("results/figures")
TABLE_DIR = Path("results/tables")

FEATURES = ["T", "D", "C", "S", "B", "H"]
HAND_PARENTS = ["D", "T", "C", "H"]
N_BINS = 10


def load_learned_parents():
    """
    Read the learned graph from the previous
    structure-learning experiment.
    """
    path = TABLE_DIR / "learned_edges.csv"

    if not path.exists():
        raise FileNotFoundError(
            "Missing learned_edges.csv. Run "
            "'python -m src.structure_learning' first."
        )

    edges = pd.read_csv(path)

    parents = sorted(
        edges.loc[
            edges["target"] == "M", "source"
        ].tolist()
    )

    print("Learned make-node parents:", parents)
    return parents


def get_predictions(train, validation):
    """
    Train each model on train_fit only and
    predict on the same validation examples.
    """
    y_train = train["M"].astype(int)
    y_val = validation["M"].astype(int)

    predictions = {}

    # 1. Constant league-average baseline
    league_rate = y_train.mean()
    predictions["League Average"] = np.full(
        len(validation), league_rate
    )

    # 2. Logistic regression
    X_train = train[FEATURES]
    X_val = validation[FEATURES]

    logistic = Pipeline([
        (
            "encoder",
            ColumnTransformer([
                (
                    "categorical",
                    OneHotEncoder(
                        handle_unknown="ignore"
                    ),
                    FEATURES
                )
            ])
        ),
        (
            "classifier",
            LogisticRegression(max_iter=1000)
        )
    ])

    logistic.fit(X_train, y_train)

    predictions["Logistic Regression"] = (
        logistic.predict_proba(X_val)[:, 1]
    )

    # 3. Naive Bayes
    nb = NaiveBayesShotModel(alpha=1.0)
    nb.fit(X_train, y_train)

    predictions["Naive Bayes"] = (
        nb.predict_proba(X_val)[:, 1]
    )

    # 4. Hand-designed BN make CPT
    hand = ConditionalMakeModel(
        parents=HAND_PARENTS,
        alpha=1.0
    )
    hand.fit(train)

    predictions["Hand-Designed BN"] = (
        hand.predict_proba(validation)
    )

    # 5. Learned BN make CPT
    learned_parents = load_learned_parents()

    learned = ConditionalMakeModel(
        parents=learned_parents,
        alpha=1.0
    )
    learned.fit(train)

    predictions["Learned BN"] = (
        learned.predict_proba(validation)
    )

    return y_val, predictions



def calculate_calibration(y_true, probabilities):
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)

    bin_idx = np.minimum(
        (probabilities * N_BINS).astype(int),
        N_BINS - 1
    )

    rows = []

    for i in range(N_BINS):
        mask = bin_idx == i

        if not mask.any():
            continue

        mean_pred = probabilities[mask].mean()
        observed = y_true[mask].mean()

        rows.append({
            "bin": i,
            "count": int(mask.sum()),
            "mean_predicted": mean_pred,
            "observed_rate": observed,
            "gap": mean_pred - observed
        })

    table = pd.DataFrame(rows)

    ece = (
        table["count"] * table["gap"].abs()
    ).sum() / len(y_true)

    return table, float(ece)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    train_raw = pd.read_csv(
        DATA_DIR / "train_fit.csv"
    )
    val_raw = pd.read_csv(
        DATA_DIR / "validation.csv"
    )

    # Match the feature preparation used in
    # the structure-learning experiment.
    train = prepare_data(train_raw)
    validation = prepare_data(val_raw)

    print("Training shots:", len(train))
    print("Validation shots:", len(validation))

    y_val, predictions = get_predictions(
        train, validation
    )

    metric_rows = []
    calibration_tables = []

    fig, ax = plt.subplots(figsize=(8, 7))

    # Perfect calibration reference
    ax.plot(
        [0, 1], [0, 1],
        "k--",
        linewidth=1.5,
        label="Perfect Calibration"
    )

    for name, probs in predictions.items():
        probs = np.asarray(probs, dtype=float)

        assert len(probs) == len(y_val)
        assert np.isfinite(probs).all()
        assert ((probs >= 0) & (probs <= 1)).all()

        print(f"\n=== {name} ===")

        metrics = evaluate_model(
            name, y_val, probs
        )

        table, ece = calculate_calibration(
            y_val, probs
        )

        print(f"ECE: {ece:.4f}")

        metric_rows.append({
            "model": name,
            **metrics,
            "ece": ece
        })

        table.insert(0, "model", name)
        calibration_tables.append(table)

        ax.plot(
            table["mean_predicted"],
            table["observed_rate"],
            marker="o",
            linewidth=2,
            label=name
        )

    ax.set_xlabel("Mean Predicted Make Probability")
    ax.set_ylabel("Observed Make Rate")
    ax.set_title(
        "NBA Shot Quality: Calibration Comparison"
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")

    plt.tight_layout()

    figure_path = (
        FIG_DIR / "calibration_comparison.png"
    )
    plt.savefig(
        figure_path, dpi=300, bbox_inches="tight"
    )
    plt.close()

    # Save comparison metrics
    metrics_df = pd.DataFrame(metric_rows)
    metrics_df = metrics_df.sort_values("log_loss")

    metrics_df.to_csv(
        TABLE_DIR / "calibration_metrics.csv",
        index=False
    )

    # Save individual calibration bins
    pd.concat(
        calibration_tables,
        ignore_index=True
    ).to_csv(
        TABLE_DIR / "calibration_bins.csv",
        index=False
    )

    # Save per-shot predictions
    prediction_df = pd.DataFrame({
        "actual": y_val.to_numpy()
    })

    for name, probs in predictions.items():
        prediction_df[name] = probs

    prediction_df.to_csv(
        TABLE_DIR / "validation_predictions.csv",
        index=False
    )

    print("\n=== FINAL COMPARISON ===")
    print(metrics_df.to_string(index=False))

    print("\nSaved files:")
    print(figure_path)
    print(TABLE_DIR / "calibration_metrics.csv")
    print(TABLE_DIR / "calibration_bins.csv")
    print(TABLE_DIR / "validation_predictions.csv")


if __name__ == "__main__":
    main()
