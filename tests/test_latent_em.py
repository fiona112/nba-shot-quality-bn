import json

import numpy as np
import pandas as pd
import pytest
from scipy.special import logsumexp

from src import latent_em
from src.latent_em import LatentShotDifficultyBN
from src.provenance import file_sha256
from src.structure_learning import FEATURES, VARIABLES, prepare_data


TRUE_P_MAKE = [0.7, 0.3]


def synthetic_shots(n=4000, seed=0):
    """
    Sample from a two-state latent-class model where each
    state strongly prefers one value of every feature.
    """
    rng = np.random.default_rng(seed)
    q = rng.integers(0, 2, size=n)

    data = {}
    for var in FEATURES:
        # 90% of the time the feature reveals the hidden state.
        reveal = rng.random(n) < 0.9
        data[var] = np.where(reveal, q, 1 - q).astype(str)

    data["M"] = (
        rng.random(n) < np.take(TRUE_P_MAKE, q)
    ).astype(int).astype(str)

    return pd.DataFrame(data)


def test_em_objective_never_decreases():
    model = LatentShotDifficultyBN(k=3, seed=1).fit(synthetic_shots())
    objective = np.array([h["objective"] for h in model.history])

    assert len(objective) > 2
    assert np.all(np.diff(objective) >= -1e-6)


def test_em_recovers_hidden_make_rates():
    model = LatentShotDifficultyBN(k=2, seed=0).fit(synthetic_shots())
    learned = sorted(model.describe_states()["p_make"])

    assert np.allclose(learned, sorted(TRUE_P_MAKE), atol=0.05)


def test_predictions_are_valid_probabilities():
    data = synthetic_shots()
    model = LatentShotDifficultyBN(k=2, seed=0).fit(data)

    # A value never seen in training is summed out, not an error.
    unseen = data.head(5).copy()
    unseen["D"] = "never-seen"

    for frame in (data, unseen):
        p = model.predict_proba(frame)
        assert np.all((p > 0) & (p < 1))


def test_prediction_does_not_require_or_use_outcomes():
    data = synthetic_shots(n=400)
    model = LatentShotDifficultyBN(k=2, max_iter=30).fit(data)
    original = model.predict_proba(data)
    flipped = data.assign(M=np.where(data["M"].eq("1"), "0", "1"))
    np.testing.assert_array_equal(original, model.predict_proba(flipped))
    np.testing.assert_array_equal(original, model.predict_proba(data.drop(columns="M")))


def test_returned_parameters_match_recorded_likelihood_at_iteration_cap():
    data = synthetic_shots(n=500)
    model = LatentShotDifficultyBN(k=3, max_iter=1).fit(data)
    actual = logsumexp(model.log_joint(model.encode(data), VARIABLES), axis=1).sum()
    assert model.log_likelihood == pytest.approx(actual, abs=1e-10)
    assert model.history[-1]["objective"] == pytest.approx(model.objective(actual), abs=1e-10)
    assert model.n_iter == 1 and not model.converged
    model.converged = True
    model.fit(data)
    assert not model.converged


def test_compression_matches_uncompressed_map_em_with_same_seed():
    data = synthetic_shots(n=700)
    compressed = LatentShotDifficultyBN(k=3, max_iter=12, tol=0, seed=8).fit(data)
    reference = LatentShotDifficultyBN(k=3, max_iter=12, tol=0, seed=8)
    reference.states = {var: sorted(data[var].unique()) for var in VARIABLES}
    codes = reference.encode(data)
    indicators = reference.one_hot(codes)
    resp = np.random.default_rng(8).dirichlet(np.ones(3), size=len(data))
    reference.m_step(indicators, resp)
    objectives = []
    for iteration in range(12):
        joint = reference.log_joint(codes, VARIABLES)
        norm = logsumexp(joint, axis=1)
        objectives.append(reference.objective(norm.sum()))
        if iteration + 1 < 12:
            reference.m_step(indicators, np.exp(joint - norm[:, None]))
    assert compressed.observed_configurations < len(data)
    np.testing.assert_allclose([item["objective"] for item in compressed.history], objectives, atol=1e-9, rtol=0)
    np.testing.assert_allclose(compressed.log_prior, reference.log_prior, atol=1e-12, rtol=0)
    for var in VARIABLES:
        np.testing.assert_allclose(compressed.log_cpts[var], reference.log_cpts[var], atol=1e-12, rtol=0)


def test_material_objective_decrease_is_not_reported_as_convergence(monkeypatch):
    model = LatentShotDifficultyBN(k=2, max_iter=3)
    original = model.objective
    calls = []

    def broken_objective(ll):
        calls.append(ll)
        return original(ll) - 10000 * len(calls)

    monkeypatch.setattr(model, "objective", broken_objective)
    with pytest.raises(RuntimeError, match="decreased"):
        model.fit(synthetic_shots(n=200))
    assert not model.converged


@pytest.mark.parametrize("configuration", [
    {"k": 0}, {"k": 1.5}, {"k": 2, "alpha": 0},
    {"k": 2, "max_iter": 0}, {"k": 2, "tol": np.nan},
])
def test_invalid_em_configuration_rejected(configuration):
    with pytest.raises(ValueError):
        LatentShotDifficultyBN(**configuration)


@pytest.fixture
def checked_em_data(tmp_path):
    frame = synthetic_shots(n=800, seed=37)
    frame.insert(0, "RAW_ROW_ID", np.arange(len(frame)))
    frame.insert(1, "GAME_ID", np.repeat(np.arange(40), 20))
    data_dir = tmp_path / "processed"
    data_dir.mkdir()
    splits = {
        "train": frame.iloc[:640], "test": frame.iloc[640:],
        "train_fit": frame.iloc[:480], "validation": frame.iloc[480:640],
    }
    report = {"schema_version": 2, "raw_rows": 800, "clean_rows": 800, "splits": {}}
    for name, subset in splits.items():
        path = data_dir / f"{name}.csv"
        subset.to_csv(path, index=False)
        report["splits"][name] = {
            "file": path.name, "shots": len(subset), "games": int(subset["GAME_ID"].nunique()),
            "sha256": file_sha256(path),
        }
    (data_dir / "preprocessing_report.json").write_text(json.dumps(report))
    return data_dir


def test_em_pipeline_selects_on_validation_and_preserves_common_test_ids(checked_em_data, tmp_path, monkeypatch):
    monkeypatch.setattr(latent_em, "DATA_DIR", checked_em_data)
    monkeypatch.setattr(latent_em, "TABLE_DIR", tmp_path / "tables")
    monkeypatch.setattr(latent_em, "FIG_DIR", tmp_path / "figures")
    monkeypatch.setattr(latent_em, "K_VALUES", [2, 3])
    monkeypatch.setattr(latent_em, "N_RESTARTS", 2)
    monkeypatch.setattr(latent_em, "MAX_ITER", 40)
    observed_training = []
    original = LatentShotDifficultyBN.fit

    def record_fit(self, data):
        observed_training.append(data.copy())
        return original(self, data)

    monkeypatch.setattr(LatentShotDifficultyBN, "fit", record_fit)
    latent_em.main()
    expected_fit = prepare_data(pd.read_csv(checked_em_data / "train_fit.csv"))
    expected_outer = prepare_data(pd.read_csv(checked_em_data / "train.csv"))
    assert len(observed_training) == 6
    for actual in observed_training[:4]:
        pd.testing.assert_frame_equal(actual, expected_fit)
    for actual in observed_training[4:]:
        pd.testing.assert_frame_equal(actual, expected_outer)
    tables = tmp_path / "tables"
    selection = pd.read_csv(tables / "latent_em_validation.csv")
    metrics = pd.read_csv(tables / "latent_em_test_metrics.csv").iloc[0]
    chosen = selection.loc[selection.val_log_loss.idxmin()]
    assert metrics.k == chosen.k
    assert metrics.validation_log_loss == pytest.approx(chosen.val_log_loss)
    details = pd.read_csv(checked_em_data / "latent_em_test_predictions.csv")
    test = pd.read_csv(checked_em_data / "test.csv")
    pd.testing.assert_frame_equal(details[["RAW_ROW_ID", "GAME_ID"]], test[["RAW_ROW_ID", "GAME_ID"]])
    np.testing.assert_array_equal(details["actual"], test["M"])
    assert details["Latent BN"].between(0, 1).all()
    bins = pd.read_csv(tables / "latent_em_calibration_bins.csv")
    assert bins["count"].sum() == len(test)
    restarts = pd.read_csv(tables / "latent_em_restarts.csv")
    assert len(restarts) == 6
    for _, group in restarts.groupby(["stage", "k"]):
        assert group.loc[group.selected_restart, "map_objective"].iloc[0] == group.map_objective.max()
    metadata = json.loads((tables / "latent_em_metadata.json").read_text())
    assert metadata["configuration"]["selected_k"] == chosen.k
    assert metadata["configuration"]["dirichlet_concentration"] == 2
    assert set(metadata["inputs"]) == {"train", "test", "train_fit", "validation"}
    for description in metadata["outputs"].values():
        assert file_sha256(description["file"]) == description["sha256"]


def test_em_loader_rejects_invalid_inner_outer_union(checked_em_data, monkeypatch):
    monkeypatch.setattr(latent_em, "DATA_DIR", checked_em_data)
    path = checked_em_data / "validation.csv"
    frame = pd.read_csv(path)
    frame["M"] = 1 - frame["M"]
    frame.to_csv(path, index=False)
    report_path = checked_em_data / "preprocessing_report.json"
    report = json.loads(report_path.read_text())
    report["splits"]["validation"]["sha256"] = file_sha256(path)
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="exactly reconstruct"):
        latent_em.load_experiment_data()
