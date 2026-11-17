"""Audit the 24-second spike and compare handling policies on training games."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.compare_distance_bins import SEEDS, compare_split
from src.preprocess import SHOT_DISTANCE_BINS, SHOT_DISTANCE_LABELS, parse_game_clock


RESULTS_DIR = Path("results/tables")


def summarize_raw_data(raw):
    game_seconds = parse_game_clock(raw["GAME_CLOCK"])
    groups = {
        "exactly_24": raw["SHOT_CLOCK"].eq(24),
        "23_to_under_24": raw["SHOT_CLOCK"].ge(23) & raw["SHOT_CLOCK"].lt(24),
        "other_recorded": raw["SHOT_CLOCK"].notna() & raw["SHOT_CLOCK"].lt(23),
        "missing_end_period": raw["SHOT_CLOCK"].isna() & game_seconds.between(0, 24),
        "missing_unexplained": raw["SHOT_CLOCK"].isna() & ~game_seconds.between(0, 24),
    }
    rows = []
    for group, mask in groups.items():
        subset = raw.loc[mask]
        rows.append({
            "clock_group": group,
            "shots": len(subset),
            "fraction_of_raw": len(subset) / len(raw),
            "median_shot_distance": subset["SHOT_DIST"].median(),
            "median_touch_time": subset["TOUCH_TIME"].median(),
            "fraction_under_4_feet": subset["SHOT_DIST"].lt(4).mean(),
            "fraction_zero_dribbles": subset["DRIBBLES"].eq(0).mean(),
            "shots_touch_time_over_10": subset["TOUCH_TIME"].gt(10).sum(),
            "shots_game_clock_at_most_24": game_seconds.loc[mask].le(24).sum(),
        })
    return pd.DataFrame(rows)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv("data/raw/shot_logs.csv", dtype={"GAME_ID": str})
    summary = summarize_raw_data(raw)
    summary.to_csv(RESULTS_DIR / "shot_clock_audit_summary.csv", index=False)
    print("Raw dataset audit:")
    print(summary.to_string(index=False), flush=True)

    # Preserve raw numeric clocks. Only vary the model's categorical S state.
    # The held-out test set is never read.
    data = pd.read_csv("data/processed/train.csv")
    results = []
    for seed in SEEDS:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
        fit_idx, validation_idx = next(splitter.split(data, groups=data["GAME_ID"]))
        assert set(data.iloc[fit_idx]["GAME_ID"]).isdisjoint(
            data.iloc[validation_idx]["GAME_ID"]
        )
        for policy in ("primary", "separate_24_state", "24_as_missing"):
            train = data.iloc[fit_idx].copy()
            validation = data.iloc[validation_idx].copy()
            if policy != "primary":
                state = "exactly24" if policy == "separate_24_state" else pd.NA
                for frame in (train, validation):
                    frame.loc[frame["SHOT_CLOCK"].eq(24), "S"] = state
            rows = compare_split(train, validation, schemes={
                policy: (SHOT_DISTANCE_BINS, SHOT_DISTANCE_LABELS)
            })
            results.extend({"seed": seed, **row} for row in rows)
        print(f"Completed clock-policy validation seed {seed}", flush=True)

    results = pd.DataFrame(results).rename(columns={"scheme": "clock_policy"})
    results.to_csv(RESULTS_DIR / "shot_clock_policy_validation.csv", index=False)
    print("\nMean metrics across five overlapping game-level validation splits:")
    print(results.groupby(["model", "clock_policy"])[
        ["log_loss", "brier_score", "accuracy"]
    ].mean().to_string())
    print("\nProduction preprocessing and raw shot clocks were not changed.")


if __name__ == "__main__":
    main()
