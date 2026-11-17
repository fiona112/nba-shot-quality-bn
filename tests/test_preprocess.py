import pandas as pd

from src import preprocess as preprocessing
from src.preprocess import parse_game_clock, preprocess, split_data, split_validation


def test_shot_distance_boundaries_and_deep_attempts():
    distances = [0, 4.9, 5, 9.9, 10, 14.9, 15, 19.9, 20, 24.9, 25, 29.9, 30, 47.2]
    data = pd.DataFrame({
        "GAME_ID": [1] * len(distances),
        "GAME_CLOCK": ["1:00"] * len(distances),
        "SHOT_RESULT": ["made"] * len(distances),
        "PTS_TYPE": [2 if distance < 22 else 3 for distance in distances],
        "SHOT_DIST": distances,
        "CLOSE_DEF_DIST": [4.0] * len(distances),
        "SHOT_CLOCK": [12.0] * len(distances),
        "DRIBBLES": [0] * len(distances),
        "TOUCH_TIME": [1.0] * len(distances),
    })

    clean = preprocess(data)

    assert clean["D"].tolist() == [
        "<5", "<5", "5-10", "5-10", "10-15", "10-15", "15-20", "15-20",
        "20-25", "20-25", "25-30", "25-30", "30+", "30+",
    ]
    # Point value remains a separate raw-data feature, even within a distance bin.
    assert clean.loc[8:9, "T"].tolist() == ["2", "3"]


def test_clock_proxy_missing_removal_and_unverified_24():
    data = pd.DataFrame({
        "GAME_ID": [1] * 13,
        "GAME_CLOCK": [
            "0:03", "0:06.5", "0:23", "0:24", "0:25", "1:00", "bad",
            "1:00", "0:05", "0:10", "0:10", "0:10", "0:00",
        ],
        "SHOT_CLOCK": [None] * 7 + [24.0, 2.0, 25.0, -1.0, "bad", None],
        "SHOT_RESULT": ["made"] * 13,
        "PTS_TYPE": [2] * 13,
        "SHOT_DIST": [3.0] * 13,
        "CLOSE_DEF_DIST": [4.0] * 13,
        "DRIBBLES": [0] * 13,
        "TOUCH_TIME": [1.0] * 13,
    })

    clean = preprocess(data)

    assert clean.index.tolist() == [0, 1, 2, 3, 7, 8, 12]
    assert clean.loc[[0, 1, 2, 3, 12], "SHOT_CLOCK"].isna().all()
    assert clean.loc[[0, 1, 2, 3, 12], "SHOT_CLOCK_EFFECTIVE"].tolist() == [
        3.0, 6.5, 23.0, 24.0, 0.0,
    ]
    assert clean.loc[[0, 1, 2, 3, 12], "SHOT_CLOCK_SOURCE"].eq("period_clock_proxy").all()
    assert clean.loc[[0, 1, 2, 3, 12], "SHOT_CLOCK_WAS_MISSING"].all()
    # A proxy equal to 24 is distinguishable from a recorded 24.
    assert not clean.loc[3, "SHOT_CLOCK_IS_24"]
    assert clean.loc[7, "SHOT_CLOCK_IS_24"]
    assert clean.loc[7, "SHOT_CLOCK_SOURCE"] == "recorded_24_unverified"
    assert clean.loc[7, "SHOT_CLOCK_EFFECTIVE"] == 24
    # An observed clock remains authoritative even near the end of the period.
    assert clean.loc[8, "SHOT_CLOCK_EFFECTIVE"] == 2
    assert clean.loc[8, "SHOT_CLOCK_SOURCE"] == "recorded"
    assert clean["S"].tolist() == ["0-4", "4-7", "22+", "22+", "22+", "0-4", "0-4"]


def test_parse_game_clock_rejects_malformed_values():
    parsed = parse_game_clock(pd.Series(["12:00", "0:23.5", " 0:24 ", "0:60", "-1:00", "13:00", None]))
    assert parsed.iloc[:3].tolist() == [720.0, 23.5, 24.0]
    assert parsed.iloc[3:].isna().all()


def test_splits_keep_games_separate_and_cover_all_shots():
    # Multiple shots per game expose accidental row-level splitting.
    data = pd.DataFrame({
        "GAME_ID": [game for game in range(50) for _ in range(3)],
        "M": [0, 1, 1] * 50,
    })
    train, test = split_data(data)
    train_fit, validation = split_validation(train)

    game_sets = [set(frame["GAME_ID"]) for frame in (train_fit, validation, test)]
    assert [len(games) for games in game_sets] == [32, 8, 10]
    assert game_sets[0].isdisjoint(game_sets[1])
    assert game_sets[0].isdisjoint(game_sets[2])
    assert game_sets[1].isdisjoint(game_sets[2])
    assert set(train_fit.index).isdisjoint(validation.index)
    assert set(train_fit.index) | set(validation.index) == set(train.index)
    assert set(train.index).isdisjoint(test.index)
    assert set(train.index) | set(test.index) == set(data.index)

    repeated_train, repeated_test = split_data(data)
    repeated_fit, repeated_validation = split_validation(repeated_train)
    for actual, repeated in zip(
        (train, test, train_fit, validation),
        (repeated_train, repeated_test, repeated_fit, repeated_validation),
    ):
        pd.testing.assert_frame_equal(actual, repeated)


def test_pipeline_logs_clock_exclusions_and_keeps_them_out_of_every_split(tmp_path, monkeypatch):
    # Each game has one observed clock, one unexplained missing clock, and one proxy.
    data = pd.DataFrame({
        "SHOT_ID": range(150),
        "GAME_ID": [game for game in range(50) for _ in range(3)],
        "GAME_CLOCK": ["1:00", "1:00", "0:10"] * 50,
        "SHOT_CLOCK": [12.0, None, None] * 50,
        "SHOT_RESULT": ["made", "missed", "made"] * 50,
        "PTS_TYPE": [2] * 150,
        "SHOT_DIST": [3.0] * 150,
        "CLOSE_DEF_DIST": [4.0] * 150,
        "DRIBBLES": [0] * 150,
        "TOUCH_TIME": [1.0] * 150,
    })
    raw_path = tmp_path / "raw.csv"
    output_dir = tmp_path / "processed"
    data.to_csv(raw_path, index=False)
    monkeypatch.setattr(preprocessing, "RAW_PATH", raw_path)
    monkeypatch.setattr(preprocessing, "OUTPUT_DIR", output_dir)

    preprocessing.main()

    excluded = pd.read_csv(output_dir / "excluded_shot_clock.csv")
    excluded_ids = set(data.loc[data.index % 3 == 1, "SHOT_ID"])
    assert set(excluded["SHOT_ID"]) == excluded_ids
    assert excluded["EXCLUDE_SHOT_CLOCK"].all()
    assert excluded["SHOT_CLOCK_EXCLUSION_REASON"].eq("missing_unexplained").all()
    assert set(excluded["RAW_ROW_ID"]) == excluded_ids
    for name in ["train", "train_fit", "validation", "test"]:
        frame = pd.read_csv(output_dir / f"{name}.csv")
        assert set(frame["SHOT_ID"]).isdisjoint(excluded_ids)
        assert not frame["EXCLUDE_SHOT_CLOCK"].any()
        assert frame["SHOT_CLOCK_EFFECTIVE"].notna().all()
    train = pd.read_csv(output_dir / "train.csv")
    test = pd.read_csv(output_dir / "test.csv")
    assert set(train["SHOT_ID"]) | set(test["SHOT_ID"]) == set(data["SHOT_ID"]) - excluded_ids
