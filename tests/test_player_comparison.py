"""Check game exclusion, shot alignment, and independent half analyses."""

import numpy as np
import pandas as pd
import pytest

from src import player_comparison as players


@pytest.fixture
def shots():
    rows = []
    for game in range(12):
        for shot in range(20):
            rows.append({
                "RAW_ROW_ID": 1000 + game * 20 + shot, "GAME_ID": game,
                "player_id": shot % 2, "player_name": "player a" if shot % 2 else "player b",
                "date": pd.Timestamp(f"2015-01-{game + 1:02d}"),
                "D": "<5" if shot % 3 else "25-30", "T": "2" if shot % 3 else "3",
                "C": "0-2", "H": "<2", "M": int(shot % 4 < 1 + game % 3),
            })
    return pd.DataFrame(rows)


def test_no_game_can_train_its_own_unique_parent_stratum(shots):
    shots["D"] = shots["GAME_ID"].astype(str)
    actual, folds = players.cross_fit_predictions(shots, n_folds=3)
    np.testing.assert_array_equal(actual["training_parent_count"], 0)
    np.testing.assert_array_equal(actual["expected"], 0.5)
    assert folds["prediction_shots"].sum() == len(shots)
    assert actual.groupby("GAME_ID")["fold"].nunique().eq(1).all()


def test_crossfit_predictions_preserve_ids_and_use_only_other_folds(shots):
    shots = shots.sample(frac=1, random_state=42)
    actual, _ = players.cross_fit_predictions(shots, n_folds=3)
    np.testing.assert_array_equal(actual["RAW_ROW_ID"], shots["RAW_ROW_ID"])
    source = shots.reset_index(drop=True)
    for _, row in actual.iloc[::17].iterrows():
        fitting = source.loc[actual["fold"].ne(row["fold"])]
        assert row["GAME_ID"] not in set(fitting["GAME_ID"])
        observation = source.loc[source["RAW_ROW_ID"].eq(row["RAW_ROW_ID"])].iloc[0]
        for parent in players.MAKE_PARENTS:
            fitting = fitting.loc[fitting[parent].eq(observation[parent])]
        assert row["training_parent_count"] == len(fitting)
        assert row["expected"] == pytest.approx((fitting["M"].sum() + 1) / (len(fitting) + 2))


def test_split_halves_do_not_train_on_each_others_outcomes(shots, monkeypatch):
    monkeypatch.setattr(players, "MIN_SHOTS_PER_HALF", 1)
    original, _, predicted, folds = players.cross_fitted_split_halves(shots, n_folds=3)
    midpoint = pd.Timestamp(original.iloc[0]["split_date"])
    changed = shots.copy()
    second = changed["date"].ge(midpoint)
    changed.loc[second, "M"] = 1 - changed.loc[second, "M"]
    _, _, altered, _ = players.cross_fitted_split_halves(changed, n_folds=3)
    first_original = predicted.loc[predicted["half"].eq("first")].set_index("RAW_ROW_ID")
    first_altered = altered.loc[altered["half"].eq("first")].set_index("RAW_ROW_ID")
    pd.testing.assert_frame_equal(first_original, first_altered)
    second_original = predicted.loc[predicted["half"].eq("second"), "expected"]
    second_altered = altered.loc[altered["half"].eq("second"), "expected"]
    assert not np.allclose(second_original, second_altered)
    assert set(predicted.loc[predicted["half"].eq("first"), "GAME_ID"]).isdisjoint(
        predicted.loc[predicted["half"].eq("second"), "GAME_ID"],
    )
    assert len(folds) == 6


def test_player_summary_is_an_exact_residual_aggregation(shots):
    predictions, _ = players.cross_fit_predictions(shots, n_folds=3)
    summary = players.summarize_players(predictions)
    for _, row in summary.iterrows():
        subset = predictions.loc[predictions["player_id"].eq(row["player_id"])]
        assert row["shots"] == len(subset)
        assert row["games"] == subset["GAME_ID"].nunique()
        assert row["fg_over_expected"] == pytest.approx((subset["M"] - subset["expected"]).mean())
        reference = (subset["M"] - subset["expected"]).sum() / np.sqrt(
            (subset["expected"] * (1 - subset["expected"])).sum(),
        )
        assert row["bernoulli_reference_z"] == pytest.approx(reference)
    assert "z_score" not in summary


def test_crossfit_rejects_duplicate_identifiers_and_too_few_games(shots):
    duplicate = shots.copy()
    duplicate.loc[1, "RAW_ROW_ID"] = duplicate.loc[0, "RAW_ROW_ID"]
    with pytest.raises(ValueError, match="unique shot"):
        players.cross_fit_predictions(duplicate, n_folds=3)
    with pytest.raises(ValueError, match="one game per fold"):
        players.cross_fit_predictions(shots, n_folds=20)
