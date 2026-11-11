
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss
)


def evaluate_model(name, y_true, y_prob):
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    y_prob = np.clip(y_prob, 1e-10, 1 - 1e-10)
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "log_loss": log_loss(
            y_true, y_prob, labels=[0, 1]
        ),
        "brier_score": brier_score_loss(
            y_true, y_prob
        ),
        "accuracy": accuracy_score(
            y_true, y_pred
        )
    }

    print(f"\n=== {name} ===")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    return metrics
