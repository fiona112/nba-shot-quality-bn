
from pathlib import Path

import pandas as pd

from src.evaluate import evaluate_model
from src.structure_learning import (
    ConditionalMakeModel,
    HAND_PARENTS,
    prepare_data,
)
from src.validation import load_learned_parents, load_validation_data, write_validation_metadata


DATA_DIR = Path("data/processed")
RESULTS_DIR = Path("results/tables")

CONFIGURATIONS = {
    "Hand BN": HAND_PARENTS,
    "Hand BN + dribbles": ["D", "T", "C", "H", "B"],
    "Hand BN + shot clock": ["D", "T", "C", "H", "S"],
    "Hand BN + both": ["D", "T", "C", "H", "B", "S"],
    "Hand BN - touch time": ["D", "T", "C"],
    "Hand BN - defender": ["D", "T", "H"],
    "Hand BN - distance": ["T", "C", "H"],
    "Hand BN - shot type": ["D", "C", "H"],
}

ALPHA = 1.0


def main():
    train_raw, val_raw, cohort = load_validation_data(DATA_DIR)
    train = prepare_data(train_raw)
    validation = prepare_data(val_raw)
    configurations = {"Learned BN": load_learned_parents(RESULTS_DIR, cohort), **CONFIGURATIONS}

    results = []

    for name, parents in configurations.items():
        print(f"\n{'=' * 55}")
        print(name)
        print("Parents:", parents)

        model = ConditionalMakeModel(
            parents=parents,
            alpha=ALPHA
        )
        model.fit(train)

        probs = model.predict_proba(validation)

        metrics = evaluate_model(
            name,
            validation["M"].astype(int),
            probs
        )

        unseen = model.coverage(validation)

        # Count theoretical configurations using
        # training-set feature state cardinalities.
        possible = 1
        for parent in parents:
            possible *= train[parent].nunique()

        observed = (
            len(model.lookup)
            if model.lookup is not None
            else 1
        )

        print("Possible parent combinations:", possible)
        print("Observed parent combinations:", observed)
        print(f"Unseen validation shots: {unseen * 100:.3f}%")

        results.append({
            "model": name,
            "training_shots": len(train),
            "validation_shots": len(validation),
            "parents": ",".join(parents),
            "num_parents": len(parents),
            "possible_combinations": possible,
            "observed_combinations": observed,
            "unseen_fraction": unseen,
            **metrics,
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("log_loss")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(
        RESULTS_DIR / "ablation_validation.csv",
        index=False
    )
    write_validation_metadata(
        RESULTS_DIR / "ablation_validation_metadata.json",
        "ablation_validation",
        cohort,
        {"alpha": ALPHA, "make_parents": configurations},
        [RESULTS_DIR / "ablation_validation.csv"],
        [RESULTS_DIR / "learned_edges.csv", RESULTS_DIR / "structure_validation_metadata.json"],
    )

    print("\n=== FINAL ABLATION COMPARISON ===")
    print(
        results_df[
            [
                "model",
                "log_loss",
                "brier_score",
                "accuracy",
                "unseen_fraction",
            ]
        ].to_string(index=False)
    )

    print("\nSaved to results/tables/ablation_validation.csv")


if __name__ == "__main__":
    main()
