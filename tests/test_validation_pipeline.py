import json

import networkx as nx
import numpy as np
import pandas as pd

from src import ablation, calibration, preprocess, structure_learning
from src.provenance import file_sha256


def test_validation_scripts_share_cohort_graph_and_probabilities(tmp_path, monkeypatch):
    """Exercise the three real entry points without relying on the NBA dataset."""
    rng = np.random.default_rng(42)
    size = 400
    raw = pd.DataFrame({
        "GAME_ID": np.repeat(np.arange(50), 8),
        "GAME_CLOCK": ["1:00"] * size,
        "SHOT_RESULT": rng.choice(["made", "missed"], size),
        "PTS_TYPE": rng.choice([2, 3], size),
        "SHOT_DIST": rng.choice([3.0, 12.0, 26.0], size),
        "CLOSE_DEF_DIST": rng.choice([1.0, 3.0, 5.0, 7.0], size),
        "SHOT_CLOCK": rng.choice([3.0, 10.0, 20.0], size),
        "DRIBBLES": rng.choice([0, 1, 5], size),
        "TOUCH_TIME": rng.choice([1.0, 4.0, 8.0], size),
    })
    raw_path = tmp_path / "raw.csv"
    raw.to_csv(raw_path, index=False)
    data_dir = tmp_path / "processed"
    tables = tmp_path / "tables"
    figures = tmp_path / "figures"
    monkeypatch.setattr(preprocess, "RAW_PATH", raw_path)
    monkeypatch.setattr(preprocess, "OUTPUT_DIR", data_dir)
    preprocess.main()

    for module in (structure_learning, ablation, calibration):
        monkeypatch.setattr(module, "DATA_DIR", data_dir)
    monkeypatch.setattr(structure_learning, "RESULTS_DIR", tables)
    monkeypatch.setattr(ablation, "RESULTS_DIR", tables)
    monkeypatch.setattr(calibration, "TABLE_DIR", tables)
    monkeypatch.setattr(calibration, "FIG_DIR", figures)
    # Deliberately differ from the legacy hard-coded D,C learned-parent set.
    monkeypatch.setattr(
        structure_learning, "learn_structure",
        lambda train: (nx.DiGraph([("H", "M")]), [("H", "M")], ["H"]),
    )
    structure_learning.main()
    ablation.main()
    calibration.main()

    structure = pd.read_csv(tables / "structure_validation.csv").set_index("model")
    ablations = pd.read_csv(tables / "ablation_validation.csv").set_index("model")
    calibrated = pd.read_csv(tables / "calibration_metrics.csv").set_index("model")
    metrics = ["log_loss", "brier_score", "accuracy"]
    for names in (
        ("Hand BN (validation)", "Hand BN", "Hand-Designed BN"),
        ("Learned BN (validation)", "Learned BN", "Learned BN"),
    ):
        expected = structure.loc[names[0], metrics].to_numpy(dtype=float)
        np.testing.assert_allclose(ablations.loc[names[1], metrics].to_numpy(dtype=float), expected, rtol=0, atol=1e-12)
        np.testing.assert_allclose(calibrated.loc[names[2], metrics].to_numpy(dtype=float), expected, rtol=0, atol=1e-12)
    assert ablations.loc["Learned BN", "parents"] == "H"

    validation = pd.read_csv(data_dir / "validation.csv")
    predictions = pd.read_csv(tables / "validation_predictions.csv")
    pd.testing.assert_frame_equal(predictions[["RAW_ROW_ID", "GAME_ID"]], validation[["RAW_ROW_ID", "GAME_ID"]])
    np.testing.assert_array_equal(predictions["actual"], validation["M"])
    probabilities = predictions.drop(columns=["RAW_ROW_ID", "GAME_ID", "actual"]).to_numpy()
    assert probabilities.shape == (len(validation), 5)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    bins = pd.read_csv(tables / "calibration_bins.csv")
    assert bins.groupby("model")["count"].sum().eq(len(validation)).all()
    assert (figures / "calibration_bin_counts.png").is_file()

    inputs = []
    for name in ("structure_validation", "ablation_validation", "calibration"):
        metadata = json.loads((tables / f"{name}_metadata.json").read_text())
        inputs.append(metadata["inputs"])
        for output in metadata["outputs"].values():
            assert file_sha256(output["file"]) == output["sha256"]
        for dependency in metadata["dependencies"].values():
            assert file_sha256(dependency["file"]) == dependency["sha256"]
    assert inputs[0] == inputs[1] == inputs[2]
