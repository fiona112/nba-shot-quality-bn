
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.evaluate import evaluate_model
from src.naive_bayes import NaiveBayesShotModel
from src.structure_learning import (
    ConditionalMakeModel,
    prepare_data,
    learn_structure,
)


DATA_DIR = Path("data/processed")
TABLE_DIR = Path("results/tables")
FIG_DIR = Path("results/figures")

FEATURES = ["T", "D", "C", "S", "B", "H"]
HAND_PARENTS = ["D", "T", "C", "H"]


def load_all_data(train=None, test=None):
    """
    Reconstruct the full cleaned dataset from
    the original game-level train/test partition.
    """
    if (train is None) != (test is None):
        raise ValueError("Supply both checked train/test frames, or neither.")
    if train is None:
        train = pd.read_csv(DATA_DIR / "train.csv")
        test = pd.read_csv(DATA_DIR / "test.csv")

    df = pd.concat(
        [train, test],
        ignore_index=True
    )

    if "GAME_DATE" in df.columns:
        df["date"] = pd.to_datetime(
            df["GAME_DATE"], errors="coerce"
        ).dt.normalize()

    elif "MATCHUP" in df.columns:
        date_text = df["MATCHUP"].astype(str).str.extract(
            r"^\s*([A-Za-z]{3}\s+\d{1,2},\s*\d{4})",
            expand=False
        )

        df["date"] = pd.to_datetime(
            date_text,
            format="%b %d, %Y",
            errors="coerce"
        )

    else:
        raise ValueError(
            "No GAME_DATE or MATCHUP column found. "
            "Preserve the original MATCHUP column "
            "during preprocessing."
        )

    if df["date"].isna().any():
        raise ValueError(
            f"Could not parse dates for "
            f"{df['date'].isna().sum()} shots."
        )

    # Each game must have exactly one date.
    assert (
        df.groupby("GAME_ID")["date"].nunique() == 1
    ).all()

    return df


def chronological_split(df):
    """
    Split by unique game dates, ensuring that
    no date or game appears in both sets.
    """
    dates = sorted(df["date"].unique())

    cutoff_index = int(0.80 * len(dates))

    train_dates = dates[:cutoff_index]
    eval_dates = dates[cutoff_index:]

    earlier = df[
        df["date"].isin(train_dates)
    ].copy()

    later = df[
        df["date"].isin(eval_dates)
    ].copy()

    assert len(earlier) > 0 and len(later) > 0

    assert set(earlier["GAME_ID"]).isdisjoint(
        set(later["GAME_ID"])
    )

    assert earlier["date"].max() < later["date"].min()

    print("\n=== CHRONOLOGICAL SPLIT ===")
    print("Earlier shots:", len(earlier))
    print("Later shots:", len(later))
    print("Earlier games:", earlier["GAME_ID"].nunique())
    print("Later games:", later["GAME_ID"].nunique())
    print("Earlier dates:",
          earlier["date"].min().date(),
          "to", earlier["date"].max().date())
    print("Later dates:",
          later["date"].min().date(),
          "to", later["date"].max().date())
    print("Earlier FG%:", round(earlier["M"].mean(), 4))
    print("Later FG%:", round(later["M"].mean(), 4))

    return earlier, later


def expected_calibration_error(y_true, probs, bins=10):
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probs, dtype=float)

    indices = np.minimum(
        (p * bins).astype(int),
        bins - 1
    )

    ece = 0.0
    calibration_rows = []

    for i in range(bins):
        mask = indices == i

        if not mask.any():
            continue

        mean_pred = float(p[mask].mean())
        observed = float(y[mask].mean())
        count = int(mask.sum())

        ece += (
            count / len(y)
        ) * abs(mean_pred - observed)

        calibration_rows.append({
            "bin": i,
            "count": count,
            "mean_predicted": mean_pred,
            "observed_rate": observed,
        })

    return ece, pd.DataFrame(calibration_rows)


def main():
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = load_all_data()
    earlier, later = chronological_split(df)

    # Separate chronology from the original random split.
    earlier.to_csv(
        DATA_DIR / "chronological_train.csv",
        index=False
    )
    later.to_csv(
        DATA_DIR / "chronological_eval.csv",
        index=False
    )

    train = prepare_data(earlier)
    evaluation = prepare_data(later)

    y_train = train["M"].astype(int)
    y_eval = evaluation["M"].astype(int)

    X_train = train[FEATURES]
    X_eval = evaluation[FEATURES]

    predictions = {}

    # 1. League-average baseline
    predictions["League Average"] = np.full(
        len(evaluation),
        y_train.mean()
    )

    # 2. Logistic regression
    logistic = Pipeline([
        (
            "encoder",
            ColumnTransformer([
                (
                    "categorical",
                    OneHotEncoder(handle_unknown="ignore"),
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
        logistic.predict_proba(X_eval)[:, 1]
    )

    # 3. Naive Bayes
    nb = NaiveBayesShotModel(alpha=1.0)
    nb.fit(X_train, y_train)

    predictions["Naive Bayes"] = (
        nb.predict_proba(X_eval)[:, 1]
    )

    # 4. Hand-designed Bayesian network
    hand = ConditionalMakeModel(
        parents=HAND_PARENTS,
        alpha=1.0
    )
    hand.fit(train)

    predictions["Hand-Designed BN"] = (
        hand.predict_proba(evaluation)
    )

    # 5. Learn a new structure using ONLY earlier games.
    print("\n=== CHRONOLOGICAL STRUCTURE LEARNING ===")

    _, learned_edges, learned_parents = (
        learn_structure(train)
    )

    print("Learned parents of M:", learned_parents)

    pd.DataFrame(
        learned_edges,
        columns=["source", "target"]
    ).to_csv(
        TABLE_DIR / "chronological_learned_edges.csv",
        index=False
    )

    learned = ConditionalMakeModel(
        parents=learned_parents,
        alpha=1.0
    )
    learned.fit(train)

    predictions["Learned BN"] = (
        learned.predict_proba(evaluation)
    )

    # Evaluate every model on the same later games.
    results = []

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(
        [0, 1], [0, 1],
        "k--", label="Perfect Calibration"
    )

    for name, probs in predictions.items():
        print(f"\n=== {name} ===")

        metrics = evaluate_model(
            name, y_eval, probs
        )

        ece, curve = expected_calibration_error(
            y_eval, probs
        )

        print("ECE:", round(ece, 4))

        results.append({
            "model": name,
            **metrics,
            "ece": ece
        })

        ax.plot(
            curve["mean_predicted"],
            curve["observed_rate"],
            marker="o",
            label=name
        )

    ax.set_title(
        "NBA Shot Quality: Chronological Evaluation"
    )
    ax.set_xlabel("Mean Predicted Make Probability")
    ax.set_ylabel("Observed Make Rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(
        FIG_DIR / "chronological_calibration.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("log_loss")

    results_df.to_csv(
        TABLE_DIR / "chronological_metrics.csv",
        index=False
    )

    print("\n=== FINAL CHRONOLOGICAL RESULTS ===")
    print(results_df.to_string(index=False))

    print("\nSaved:")
    print(TABLE_DIR / "chronological_metrics.csv")
    print(TABLE_DIR / "chronological_learned_edges.csv")
    print(FIG_DIR / "chronological_calibration.png")


if __name__ == "__main__":
    main()
