
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluate import evaluate_model


FEATURES = ["T", "D", "C", "S", "B", "H"]
DATA_DIR = Path("data/processed")
ALPHA = 1.0


class NaiveBayesShotModel:
    """
    Discrete Naive Bayes classifier for NBA shot outcomes.

    Estimates:
        P(M) and P(feature | M)

    Uses Laplace smoothing (alpha=1 by default).
    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.classes = [0, 1]
        self.log_prior = {}
        self.log_cpts = {}
        self.unknown_log_probs = {}

    def fit(self, X, y):
        X = X.copy().fillna("missing").astype(str)
        y = pd.Series(np.asarray(y), index=X.index)

        n = len(y)

        # Learn the prior P(M)
        for m in self.classes:
            count = int((y == m).sum())
            prior = (count + self.alpha) / (
                n + self.alpha * len(self.classes)
            )
            self.log_prior[m] = np.log(prior)

        # Learn feature CPTs: P(X_i | M)
        for feature in FEATURES:
            self.log_cpts[feature] = {}
            self.unknown_log_probs[feature] = {}

            values = sorted(X[feature].unique())
            k = len(values)

            for m in self.classes:
                subset = X.loc[y == m, feature]
                counts = subset.value_counts()
                total = len(subset)

                # Reserve one additional state for unseen values
                denominator = total + self.alpha * (k + 1)

                probs = {
                    value: (counts.get(value, 0) + self.alpha)
                           / denominator
                    for value in values
                }

                self.log_cpts[feature][m] = {
                    value: np.log(prob)
                    for value, prob in probs.items()
                }

                self.unknown_log_probs[feature][m] = np.log(
                    self.alpha / denominator
                )

        return self

    def predict_proba(self, X):
        X = X.copy().fillna("missing").astype(str)

        scores = np.zeros((len(X), 2))

        for class_idx, m in enumerate(self.classes):
            scores[:, class_idx] = self.log_prior[m]

            for feature in FEATURES:
                cpt = self.log_cpts[feature][m]
                default = self.unknown_log_probs[feature][m]

                scores[:, class_idx] += (
                    X[feature]
                    .map(cpt)
                    .fillna(default)
                    .to_numpy(dtype=float)
                )

        # Normalize log scores into probabilities
        log_norm = np.logaddexp(
            scores[:, 0], scores[:, 1]
        )

        return np.exp(scores - log_norm[:, None])

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return (probabilities[:, 1] >= 0.5).astype(int)


def main():
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")

    X_train = train[FEATURES]
    X_test = test[FEATURES]

    y_train = train["M"].astype(int)
    y_test = test["M"].astype(int)

    model = NaiveBayesShotModel(alpha=ALPHA)
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = evaluate_model(
        "Naive Bayes", y_test, probabilities
    )

    results_dir = Path("results/tables")
    results_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([
        {"model": "Naive Bayes", **metrics}
    ]).to_csv(
        results_dir / "naive_bayes_metrics.csv",
        index=False
    )

    print("\nExample predicted probabilities:")
    for p in probabilities[:10]:
        print(f"P(M=1 | features) = {p:.4f}")

    print("\nNaive Bayes results saved!")


if __name__ == "__main__":
    main()
