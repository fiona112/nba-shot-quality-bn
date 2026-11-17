"""Compare feature handling on identical cleaned training and test shots."""

from pathlib import Path

import pandas as pd

from src.compare_distance_bins import SCHEMES, compare_split


OUTPUT_PATH = Path("results/tables/preprocessing_comparison_test.csv")


def main():
    train = pd.read_csv("data/processed/train.csv")
    test = pd.read_csv("data/processed/test.csv")
    assert not train["EXCLUDE_SHOT_CLOCK"].any()
    assert not test["EXCLUDE_SHOT_CLOCK"].any()
    assert set(train["GAME_ID"]).isdisjoint(test["GAME_ID"])

    experiments = [
        ("original_buckets_missing_state", "original", False),
        ("new_buckets_missing_state", "five_foot_30plus", False),
        ("new_buckets_period_proxy", "five_foot_30plus", True),
    ]
    results = []
    for configuration, scheme, use_period_proxy in experiments:
        fit = train.copy()
        held_out = test.copy()
        if not use_period_proxy:
            for frame in (fit, held_out):
                frame.loc[frame["SHOT_CLOCK_WAS_MISSING"], "S"] = pd.NA
        rows = compare_split(fit, held_out, schemes={configuration: SCHEMES[scheme]})
        for row in rows:
            row["clock_preprocessing"] = (
                "same_clean_rows_period_proxy" if use_period_proxy
                else "same_clean_rows_missing_state"
            )
        results.extend(rows)
        print(f"Completed {configuration}", flush=True)

    results = pd.DataFrame(results).rename(columns={
        "scheme": "configuration", "validation_shots": "test_shots"
    })
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_PATH, index=False)
    print("\nAll configurations use identical cleaned training/test shots.")
    print("Unexplained missing-clock exclusions are respected in every configuration.")
    print(results.to_string(index=False))
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
