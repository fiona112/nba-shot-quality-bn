
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


RAW_PATH = Path("data/raw/shot_logs.csv")
OUTPUT_DIR = Path("data/processed")

FEATURES = ["T", "D", "C", "S", "B", "H"]
REQUIRED = [
    "GAME_ID", "SHOT_RESULT", "PTS_TYPE",
    "SHOT_DIST", "CLOSE_DEF_DIST",
    "SHOT_CLOCK", "DRIBBLES", "TOUCH_TIME"
]


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
    df = df.copy()

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

    # Keep rows with valid targets and required features.
    # Missing shot clock is allowed.
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

    # Leave shot clocks at exactly 24.0 untouched
    # until the data audit is reviewed.
    valid &= df["SHOT_CLOCK"].isna() | (
        df["SHOT_CLOCK"].between(0, 24)
    )

    print("\nRows removed:", (~valid).sum())
    df = df.loc[valid].copy()

    # Shot type
    df["T"] = df["PTS_TYPE"].astype(int).astype(str)

    # Shot distance (feet)
    df["D"] = pd.cut(
        df["SHOT_DIST"],
        bins=[-np.inf, 4, 10, 16, 22, 26, np.inf],
        labels=["<4", "4-10", "10-16", "16-22",
                "22-26", "26+"],
        right=False
    )

    # Closest defender distance (feet)
    df["C"] = pd.cut(
        df["CLOSE_DEF_DIST"],
        bins=[-np.inf, 2, 4, 6, np.inf],
        labels=["0-2", "2-4", "4-6", "6+"],
        right=False
    )

    # Shot clock (seconds)
    df["S"] = pd.cut(
        df["SHOT_CLOCK"],
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


def main():
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Dataset missing: {RAW_PATH}"
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

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(OUTPUT_DIR / "train.csv", index=False)
    test.to_csv(OUTPUT_DIR / "test.csv", index=False)

    print("\nSaved processed data successfully!")


if __name__ == "__main__":
    main()
