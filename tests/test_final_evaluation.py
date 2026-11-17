import json

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from src import final_evaluation as final
from src.chronological_evaluation import chronological_split, load_all_data
from src.provenance import file_sha256
from src.structure_learning import prepare_data


@pytest.fixture
def checked_data(tmp_path):
    rng = np.random.default_rng(42)
    n = 240
    frame = pd.DataFrame({
        "RAW_ROW_ID": np.arange(n), "GAME_ID": np.repeat(np.arange(30), 8),
        "GAME_DATE": np.repeat(pd.date_range("2024-01-01", periods=30).astype(str), 8),
        "T": rng.choice(["2", "3"], n), "D": rng.choice(["<5", "10-15", "25-30"], n),
        "C": rng.choice(["0-2", "2-4", "6+"], n),
        "S": rng.choice(["0-4", "7-15", "22+"], n),
        "B": rng.choice(["0", "1", "7+"], n), "H": rng.choice(["<2", "2-6", "6+"], n),
        "M": rng.integers(0, 2, n),
    })
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    train, test = frame.iloc[:160].copy(), frame.iloc[160:].copy()
    report = {"schema_version": 2, "raw_rows": n, "clean_rows": n, "splits": {}}
    for name, subset in (("train", train), ("test", test)):
        path = data_dir / f"{name}.csv"
        subset.to_csv(path, index=False)
        report["splits"][name] = {
            "shots": len(subset), "games": int(subset["GAME_ID"].nunique()),
            "file": path.name, "sha256": file_sha256(path),
        }
    (data_dir / "preprocessing_report.json").write_text(json.dumps(report))
    return data_dir


def test_final_entry_point_fits_training_only_and_compares_identical_shots(checked_data, tmp_path, monkeypatch):
    tables, figures = tmp_path / "tables", tmp_path / "figures"
    monkeypatch.setattr(final, "DATA_DIR", checked_data)
    monkeypatch.setattr(final, "TABLE_DIR", tables)
    monkeypatch.setattr(final, "FIG_DIR", figures)
    learned_inputs = []

    def learn(train):
        learned_inputs.append(train.copy())
        return nx.DiGraph([("D", "M")]), [("D", "M")], ["D"]

    monkeypatch.setattr(final, "learn_structure", learn)
    train, test, _ = final.load_final_data(checked_data)
    combined = load_all_data(train, test)
    earlier, later = chronological_split(combined)
    final.main()

    # Exact frames, including targets, prove structure learning never sees the heldout labels.
    pd.testing.assert_frame_equal(learned_inputs[0], prepare_data(train))
    pd.testing.assert_frame_equal(learned_inputs[1], prepare_data(earlier))
    for prefix, heldout in (("final_test", test), ("chronological", later)):
        metrics = pd.read_csv(tables / f"{prefix}_metrics.csv")
        predictions = pd.read_csv(checked_data / f"{prefix}_predictions.csv")
        assert set(metrics["model"]) == set(final.MODEL_NAMES)
        assert metrics["evaluation_shots"].eq(len(heldout)).all()
        assert metrics["evaluation_games"].eq(heldout["GAME_ID"].nunique()).all()
        pd.testing.assert_frame_equal(
            predictions[["RAW_ROW_ID", "GAME_ID"]], heldout[["RAW_ROW_ID", "GAME_ID"]].reset_index(drop=True),
        )
        np.testing.assert_array_equal(predictions["actual"], heldout["M"])
        probabilities = predictions[list(final.MODEL_NAMES)].to_numpy()
        assert np.isfinite(probabilities).all()
        assert ((probabilities >= 0) & (probabilities <= 1)).all()
        bins = pd.read_csv(tables / f"{prefix}_calibration_bins.csv")
        assert bins.groupby("model")["count"].sum().eq(len(heldout)).all()
        assert metrics.set_index("model").loc["Learned BN", "parents"] == "D"
        assert (figures / f"{prefix}_calibration.png").is_file()
        metadata = json.loads((tables / f"{prefix}_metadata.json").read_text())
        assert metadata["configuration"]["model_selection_on_evaluation"] is False
        assert metadata["configuration"]["primary_parents"] == ["D", "T", "C", "H"]
        assert metadata["configuration"]["learned_parents"] == ["D"]
        paired = pd.read_csv(tables / f"{prefix}_paired_comparisons.csv")
        assert set(paired["comparator"]) == set(final.MODEL_NAMES) - {"Hand-Designed BN"}
        assert paired["shots"].eq(len(heldout)).all()
        assert paired["games"].eq(heldout["GAME_ID"].nunique()).all()
        assert paired["bootstrap_replicates"].eq(1000).all()
        assert metadata["configuration"]["paired_comparisons"]["refit_models_in_bootstrap"] is False
        for output in metadata["outputs"].values():
            assert file_sha256(output["file"]) == output["sha256"]
        for source in metadata["inputs"].values():
            assert file_sha256(source["file"]) == source["sha256"]
    chronology = json.loads((tables / "chronological_metadata.json").read_text())
    assert chronology["cohorts"]["training"]["date_max"] < chronology["cohorts"]["evaluation"]["date_min"]
    assert sum(chronology["original_random_test_overlap"].values()) == len(test)


def test_heldout_target_changes_cannot_change_any_predictions(checked_data):
    train_raw, test_raw, _ = final.load_final_data(checked_data)
    train, test = prepare_data(train_raw), prepare_data(test_raw)
    _, original = final.get_predictions(train, test, ["D"])
    test["M"] = np.where(test["M"].eq("1"), "0", "1")
    _, flipped = final.get_predictions(train, test, ["D"])
    for name in final.MODEL_NAMES:
        np.testing.assert_array_equal(original[name], flipped[name])


@pytest.mark.parametrize("invalid", [
    "hash", "duplicate_rows", "overlap_games", "overlap_rows", "invalid_row_id",
    "count", "union", "missing_feature", "invalid_target",
])
def test_final_loader_rejects_stale_or_invalid_cohorts(checked_data, invalid):
    path = checked_data / "test.csv"
    frame = pd.read_csv(path)
    report_path = checked_data / "preprocessing_report.json"
    report = json.loads(report_path.read_text())
    if invalid == "duplicate_rows":
        frame.loc[1, "RAW_ROW_ID"] = frame.loc[0, "RAW_ROW_ID"]
    elif invalid == "overlap_games":
        frame["GAME_ID"] = frame["GAME_ID"].replace({20: 0})
    elif invalid == "overlap_rows":
        frame.loc[0, "RAW_ROW_ID"] = 0
    elif invalid == "invalid_row_id":
        frame.loc[0, "RAW_ROW_ID"] = -1
    elif invalid == "missing_feature":
        frame = frame.drop(columns="D")
    elif invalid == "invalid_target":
        frame.loc[0, "M"] = 7
    elif invalid == "count":
        report["splits"]["test"]["shots"] += 1
    elif invalid == "union":
        report["clean_rows"] += 1
    frame.to_csv(path, index=False)
    if invalid == "hash":
        with path.open("a") as stream:
            stream.write("\n")
    else:
        report["splits"]["test"]["sha256"] = file_sha256(path)
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        final.load_final_data(checked_data)


def test_checked_chronology_uses_supplied_frames_and_dates(checked_data, monkeypatch):
    train, test, _ = final.load_final_data(checked_data)

    def unexpected_read(*args, **kwargs):
        raise AssertionError("Checked chronology must not reread files.")

    monkeypatch.setattr(pd, "read_csv", unexpected_read)
    combined = load_all_data(train, test)
    earlier, later = chronological_split(combined)
    assert earlier["date"].max() < later["date"].min()
    assert set(earlier["RAW_ROW_ID"]).isdisjoint(later["RAW_ROW_ID"])
    assert set(earlier["GAME_ID"]).isdisjoint(later["GAME_ID"])
    assert len(earlier) + len(later) == len(combined)
    with pytest.raises(ValueError, match="both checked"):
        load_all_data(train)


def test_frozen_protocol_rejects_configuration_drift(monkeypatch):
    monkeypatch.setattr(final, "ALPHA", 3.0)
    with pytest.raises(ValueError, match="frozen"):
        final.frozen_protocol()


def test_paired_comparisons_preserve_shots_whole_games_and_positive_hand_advantage(monkeypatch):
    y = np.array([0, 1, 0, 1, 1, 0])
    # Unequal game sizes and unsorted row IDs expose row/cluster alignment errors.
    games = np.array([7, 7, 4, 9, 4, 4])
    correct_hand = np.array([0.8, 0.95, 0.7, 0.85, 0.9, 0.6])
    per_shot = pd.DataFrame({
        "RAW_ROW_ID": [50, 3, 88, 12, 6, 101], "GAME_ID": games, "actual": y,
        "Hand-Designed BN": np.where(y, correct_hand, 1 - correct_hand),
    }, index=[9, 3, 7, 1, 8, 2])
    comparators = [name for name in final.MODEL_NAMES if name != "Hand-Designed BN"]
    for index, name in enumerate(comparators):
        correct = 0.5 + index * 0.02
        per_shot[name] = np.where(y, correct, 1 - correct)
    original = final.paired_game_bootstrap
    calls = []

    def record(differences, actual_games, replicates, seed):
        calls.append((np.asarray(differences), np.asarray(actual_games)))
        assert replicates == 1000 and seed == 20261117
        return original(differences, actual_games, replicates=replicates, seed=seed)

    monkeypatch.setattr(final, "paired_game_bootstrap", record)
    result = final.paired_comparisons(per_shot)
    assert list(result["comparator"]) == comparators
    assert result["log_loss_improvement"].gt(0).all()
    primary_loss = final.log_losses(y, per_shot["Hand-Designed BN"])
    for name, (differences, actual_games), (_, row) in zip(comparators, calls, result.iterrows()):
        np.testing.assert_array_equal(actual_games, games)
        expected = final.log_losses(y, per_shot[name]) - primary_loss
        np.testing.assert_allclose(differences, expected)
        assert row["shots"] == 6 and row["games"] == 3
        assert row["log_loss_improvement"] == pytest.approx(expected.mean())
        # Independently reconstruct whole-game sampling with unequal shot counts.
        grouped = pd.DataFrame({"game": games, "loss": expected}).groupby("game")["loss"].agg(["sum", "count"])
        draw = np.random.default_rng(20261117).integers(0, 3, size=(1000, 3))
        estimates = grouped["sum"].to_numpy()[draw].sum(axis=1) / grouped["count"].to_numpy()[draw].sum(axis=1)
        np.testing.assert_allclose([row["ci95_low"], row["ci95_high"]], np.quantile(estimates, [0.025, 0.975]))
