
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


RAW_PATH = Path("data/raw/shot_logs.csv")
OUTPUT_DIR = Path("data/processed")

FEATURES = ["T", "D", "C", "S", "B", "H"]
SHOT_DISTANCE_BINS = [0, 5, 10, 15, 20, 25, 30, np.inf]
SHOT_DISTANCE_LABELS = ["<5", "5-10", "10-15", "15-20", "20-25", "25-30", "30+"]
REQUIRED = [
    "GAME_ID", "GAME_CLOCK", "SHOT_RESULT", "PTS_TYPE",
    "SHOT_DIST", "CLOSE_DEF_DIST",
    "SHOT_CLOCK", "DRIBBLES", "TOUCH_TIME"
]


def parse_game_clock(values):
    """Parse M:SS or M:SS.s; malformed clocks remain unknown."""
    parts = values.astype("string").str.strip().str.extract(
        r"^(\d{1,2}):(\d{1,2}(?:\.\d+)?)$"
    )
    minutes = pd.to_numeric(parts[0], errors="coerce")
    seconds = pd.to_numeric(parts[1], errors="coerce")
    total = minutes * 60 + seconds
    return total.where(seconds.ge(0) & seconds.lt(60) & total.le(720))


def prepare_shot_clock(df):
    """Preserve recorded clocks and create a documented modeling proxy."""
    df = df.copy()
    raw = df["SHOT_CLOCK"]
    clock = pd.to_numeric(raw, errors="coerce")
    df["GAME_CLOCK_SECONDS"] = parse_game_clock(df["GAME_CLOCK"])
    df["SHOT_CLOCK_WAS_MISSING"] = raw.isna()
    df["SHOT_CLOCK_IS_24"] = clock.eq(24)

    recorded = clock.between(0, 24)
    # This window permits a clock-off explanation but cannot prove it.
    period_proxy = raw.isna() & df["GAME_CLOCK_SECONDS"].between(0, 24)
    df["SHOT_CLOCK_EFFECTIVE"] = clock.where(recorded)
    df.loc[period_proxy, "SHOT_CLOCK_EFFECTIVE"] = df.loc[
        period_proxy, "GAME_CLOCK_SECONDS"
    ]
    df["SHOT_CLOCK_SOURCE"] = pd.Series("invalid_recorded", index=df.index)
    df.loc[raw.isna(), "SHOT_CLOCK_SOURCE"] = "missing_unexplained"
    df.loc[recorded, "SHOT_CLOCK_SOURCE"] = "recorded"
    df.loc[period_proxy, "SHOT_CLOCK_SOURCE"] = "period_clock_proxy"
    df.loc[df["SHOT_CLOCK_IS_24"], "SHOT_CLOCK_SOURCE"] = "recorded_24_unverified"
    df["EXCLUDE_SHOT_CLOCK"] = df["SHOT_CLOCK_EFFECTIVE"].isna()
    df["SHOT_CLOCK_EXCLUSION_REASON"] = df["SHOT_CLOCK_SOURCE"].where(
        df["EXCLUDE_SHOT_CLOCK"], pd.NA
    )
    return df


def audit_data(df):
    print("\n=== DATA AUDIT ===")
    print("Original shape:", df.shape)
    print("\nColumns:", df.columns.tolist())

    print("\nMissing values:")
    print(df.isna().sum().sort_values(ascending=False))

    print("\nShot results:")
    print(df["SHOT_RESULT"].value_counts(dropna=False))

    print("\nShot-clock analysis:")
    clock = pd.to_numeric(df["SHOT_CLOCK"], errors="coerce")
    print("Missing shot clocks:", clock.isna().sum())
    print("Exactly 24.0:", (clock == 24.0).sum())
    print("Percent exactly 24.0:",
          round(100 * (clock == 24.0).mean(), 2))

    print("\nMost common shot-clock values:")
    print(clock.value_counts().head(15))

    print("\nShot clocks near 24 seconds:")
    print(
        clock[(clock >= 22) & (clock <= 24)]
        .value_counts()
        .sort_index()
        .tail(25)
    )

    if "GAME_CLOCK" in df.columns:
        game_seconds = parse_game_clock(df["GAME_CLOCK"])
        print("Missing clocks with game clock <=24:",
              (clock.isna() & game_seconds.between(0, 24)).sum())
        print("Missing clocks outside that window or with invalid game clock:",
              (clock.isna() & ~game_seconds.between(0, 24)).sum())
        print("\nGAME_CLOCK values for missing shot clocks:")
        print(
            df.loc[clock.isna(), "GAME_CLOCK"]
            .value_counts()
            .head(15)
        )

        print("\nGAME_CLOCK values for 24.0 shot clocks:")
        print(
            df.loc[clock == 24.0, "GAME_CLOCK"]
            .value_counts()
            .head(15)
        )

    print("\nNumeric feature summaries:")
    columns = [
        "SHOT_DIST", "CLOSE_DEF_DIST",
        "SHOT_CLOCK", "DRIBBLES", "TOUCH_TIME"
    ]
    print(df[columns].describe())


def preprocess(df):
    df = prepare_shot_clock(df)

    # Standardize names and numeric inputs
    numeric_cols = [
        "PTS_TYPE", "SHOT_DIST", "CLOSE_DEF_DIST",
        "SHOT_CLOCK", "DRIBBLES", "TOUCH_TIME"
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["M"] = (
        df["SHOT_RESULT"]
        .astype("string")
        .str.strip()
        .str.lower()
        .map({"made": 1, "missed": 0})
    )

    # Keep valid targets/features and recorded or end-of-period proxy clocks.
    required_nonmissing = [
        "GAME_ID", "M", "PTS_TYPE", "SHOT_DIST",
        "CLOSE_DEF_DIST", "DRIBBLES", "TOUCH_TIME"
    ]
    valid = df[required_nonmissing].notna().all(axis=1)

    valid &= df["PTS_TYPE"].isin([2, 3])
    valid &= df["SHOT_DIST"].ge(0)
    valid &= df["CLOSE_DEF_DIST"].ge(0)
    valid &= df["DRIBBLES"].ge(0)
    valid &= df["TOUCH_TIME"].ge(0)

    # Unexplained missing clocks and invalid recorded clocks have no proxy.
    valid &= ~df["EXCLUDE_SHOT_CLOCK"]

    print("\nRows removed:", (~valid).sum())
    df = df.loc[valid].copy()
    print("\nShot-clock sources retained:")
    print(df["SHOT_CLOCK_SOURCE"].value_counts())

    # Shot type
    df["T"] = df["PTS_TYPE"].astype(int).astype(str)

    # Five-foot distance ranges, with 30+ feet kept separate for deep attempts.
    df["D"] = pd.cut(
        df["SHOT_DIST"],
        bins=SHOT_DISTANCE_BINS,
        labels=SHOT_DISTANCE_LABELS,
        right=False
    )

    # Closest defender distance (feet)
    df["C"] = pd.cut(
        df["CLOSE_DEF_DIST"],
        bins=[-np.inf, 2, 4, 6, np.inf],
        labels=["0-2", "2-4", "4-6", "6+"],
        right=False
    )

    # Use game-clock time as a flagged proxy only for missing end-period clocks.
    df["S"] = pd.cut(
        df["SHOT_CLOCK_EFFECTIVE"],
        bins=[-np.inf, 4, 7, 15, 18, 22, np.inf],
        labels=["0-4", "4-7", "7-15",
                "15-18", "18-22", "22+"],
        right=False
    )

    # Dribbles
    df["B"] = pd.cut(
        df["DRIBBLES"],
        bins=[-np.inf, 1, 2, 3, 7, np.inf],
        labels=["0", "1", "2", "3-6", "7+"],
        right=False
    )

    # Touch time (seconds)
    df["H"] = pd.cut(
        df["TOUCH_TIME"],
        bins=[-np.inf, 2, 6, np.inf],
        labels=["<2", "2-6", "6+"],
        right=False
    )

    print("\n=== BIN COUNTS ===")
    for col in FEATURES:
        print(f"\n{col}:")
        print(df[col].value_counts(dropna=False))

    # Convert categories to strings, retaining missing values
    for col in FEATURES:
        df[col] = df[col].astype("string")

    df["M"] = df["M"].astype(int)
    return df


def split_data(df):
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42
    )

    train_idx, test_idx = next(
        splitter.split(df, groups=df["GAME_ID"])
    )

    train = df.iloc[train_idx].copy()
    test = df.iloc[test_idx].copy()

    assert set(train["GAME_ID"]).isdisjoint(
        set(test["GAME_ID"])
    )

    print("\n=== DATA SPLIT ===")
    print("Training shots:", len(train))
    print("Testing shots:", len(test))
    print("Training FG%:", round(train["M"].mean(), 4))
    print("Testing FG%:", round(test["M"].mean(), 4))

    return train, test


def split_validation(train):
    """Reserve 20% of training games for model selection, keeping test untouched."""
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=123
    )
    fit_idx, validation_idx = next(
        splitter.split(train, groups=train["GAME_ID"])
    )
    train_fit = train.iloc[fit_idx].copy()
    validation = train.iloc[validation_idx].copy()

    assert set(train_fit["GAME_ID"]).isdisjoint(
        set(validation["GAME_ID"])
    )
    print("Training-fit shots:", len(train_fit))
    print("Validation shots:", len(validation))
    return train_fit, validation


def main():
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Dataset missing: {RAW_PATH}. Download and extract shot_logs.csv "
            "from https://www.kaggle.com/datasets/dansbecker/nba-shot-logs "
            "into data/raw/ first."
        )

    df = pd.read_csv(RAW_PATH)

    missing_cols = set(REQUIRED) - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Missing required columns: {missing_cols}"
        )

    audit_data(df)
    clean_df = preprocess(df)
    train, test = split_data(clean_df)
    train_fit, validation = split_validation(train)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(OUTPUT_DIR / "train.csv", index=False)
    test.to_csv(OUTPUT_DIR / "test.csv", index=False)
    train_fit.to_csv(OUTPUT_DIR / "train_fit.csv", index=False)
    validation.to_csv(OUTPUT_DIR / "validation.csv", index=False)

    clock_audit = prepare_shot_clock(df)
    clock_excluded = clock_audit.loc[clock_audit["EXCLUDE_SHOT_CLOCK"]].copy()
    clock_excluded.insert(0, "RAW_ROW_ID", clock_excluded.index)
    clock_excluded.to_csv(OUTPUT_DIR / "excluded_shot_clock.csv", index=False)
    clock_sources = clock_audit["SHOT_CLOCK_SOURCE"].value_counts()
    report = {
        "raw_rows": len(df),
        "clean_rows": len(clean_df),
        "removed_rows_total": len(df) - len(clean_df),
        "excluded_shot_clock_rows": len(clock_excluded),
        "excluded_shot_clock_file": "excluded_shot_clock.csv",
        "raw_shot_clock_sources": {name: int(count) for name, count in clock_sources.items()},
        "retained_shot_clock_sources": {
            name: int(count)
            for name, count in clean_df["SHOT_CLOCK_SOURCE"].value_counts().items()
        },
        "missing_clock_policy": "game_clock_proxy_if_0_to_24_seconds_else_drop",
        "recorded_24_policy": "keep_and_flag_unverified",
        "splits": {
            name: {"shots": len(frame), "games": int(frame["GAME_ID"].nunique())}
            for name, frame in {
                "train": train, "test": test,
                "train_fit": train_fit, "validation": validation,
            }.items()
        },
    }
    (OUTPUT_DIR / "preprocessing_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )

    print("\nSaved processed data successfully!")


if __name__ == "__main__":
    main()
