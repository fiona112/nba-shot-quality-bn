
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.evaluate import evaluate_model


DATA_DIR = Path("data/processed")
FEATURES = ["T", "D", "C", "S", "B", "H"]


def load_data():
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")

    X_train = train[FEATURES].fillna("missing").astype(str)
    X_test = test[FEATURES].fillna("missing").astype(str)

    y_train = train["M"].astype(int)
    y_test = test["M"].astype(int)

    return X_train, X_test, y_train, y_test


def main():
    X_train, X_test, y_train, y_test = load_data()

    # Baseline 1: constant training-set FG%
    league_rate = y_train.mean()
    constant_probs = np.full(len(y_test), league_rate)

    results = []
    metrics = evaluate_model(
        "League Average", y_test, constant_probs
    )
    results.append({"model": "League Average", **metrics})

    # Baseline 2: logistic regression
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

    logistic_probs = logistic.predict_proba(X_test)[:, 1]

    metrics = evaluate_model(
        "Logistic Regression", y_test, logistic_probs
    )
    results.append({"model": "Logistic Regression", **metrics})

    # Save metrics
    results_dir = Path("results/tables")
    results_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(results).to_csv(
        results_dir / "baseline_metrics.csv",
        index=False
    )

    print("\nBaseline metrics saved!")


if __name__ == "__main__":
    main()
