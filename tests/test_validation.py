"""Protect validation cohorts and their learned-graph dependencies."""

from copy import deepcopy
import json

import pandas as pd
import pytest

from src.provenance import file_sha256
from src.validation import load_learned_parents, load_validation_data


def make_cohort(tmp_path, overlap=None):
    frames = {
        "train_fit": pd.DataFrame({
            "RAW_ROW_ID": [0, 1], "GAME_ID": [10, 10],
            "T": [2, 3], "D": ["<5", "25-30"], "C": ["0-2", "6+"],
            "S": ["7-15", "4-7"], "B": ["0", "1"], "H": ["<2", "2-6"],
            "M": [1, 0],
        }),
        "validation": pd.DataFrame({
            "RAW_ROW_ID": [2, 3], "GAME_ID": [20, 20],
            "T": [2, 3], "D": ["<5", "25-30"], "C": ["0-2", "6+"],
            "S": ["7-15", "4-7"], "B": ["0", "1"], "H": ["<2", "2-6"],
            "M": [0, 1],
        }),
    }
    if overlap is not None:
        frames["validation"].loc[0, overlap] = frames["train_fit"].loc[0, overlap]
    report = {"schema_version": 2, "raw_input": {"sha256": "raw-data-hash"}, "splits": {}}
    for name, frame in frames.items():
        path = tmp_path / f"{name}.csv"
        frame.to_csv(path, index=False)
        report["splits"][name] = {
            "sha256": file_sha256(path), "shots": len(frame),
            "games": int(frame["GAME_ID"].nunique()),
        }
    (tmp_path / "preprocessing_report.json").write_text(json.dumps(report))
    return frames


def save_graph(tmp_path, cohort, edges):
    path = tmp_path / "learned_edges.csv"
    pd.DataFrame(edges, columns=["source", "target"]).to_csv(path, index=False)
    metadata = {
        "schema_version": 1,
        "experiment": "structure_validation",
        **deepcopy(cohort),
        "outputs": {"learned_edges.csv": {"sha256": file_sha256(path)}},
    }
    (tmp_path / "structure_validation_metadata.json").write_text(json.dumps(metadata))
    return metadata


def test_validation_inputs_match_checked_files_and_preserve_identifiers(tmp_path):
    frames = make_cohort(tmp_path)
    train, validation, cohort = load_validation_data(tmp_path)
    for name, actual in [("train_fit", train), ("validation", validation)]:
        assert actual["RAW_ROW_ID"].tolist() == frames[name]["RAW_ROW_ID"].tolist()
        assert cohort["inputs"][name]["sha256"] == file_sha256(tmp_path / f"{name}.csv")
        assert cohort["inputs"][name]["shots"] == 2
    assert cohort["preprocessing_report_sha256"] == file_sha256(tmp_path / "preprocessing_report.json")


def test_modified_split_is_rejected_before_models_run(tmp_path):
    make_cohort(tmp_path)
    path = tmp_path / "validation.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "M"] = 1
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="does not match the preprocessing report"):
        load_validation_data(tmp_path)


@pytest.mark.parametrize("column", ["GAME_ID", "RAW_ROW_ID"])
def test_overlapping_splits_fail_even_when_report_hashes_and_counts_match(tmp_path, column):
    make_cohort(tmp_path, overlap=column)
    with pytest.raises(ValueError, match=f"overlap in {column}"):
        load_validation_data(tmp_path)


def test_graph_requires_metadata_beside_edges(tmp_path):
    make_cohort(tmp_path)
    _, _, cohort = load_validation_data(tmp_path)
    pd.DataFrame([("D", "M")], columns=["source", "target"]).to_csv(
        tmp_path / "learned_edges.csv", index=False,
    )
    with pytest.raises(FileNotFoundError, match="Missing graph/provenance"):
        load_learned_parents(tmp_path, cohort)


@pytest.mark.parametrize("contents", ["{", "{}", "null"])
def test_incomplete_graph_metadata_has_an_actionable_error(tmp_path, contents):
    make_cohort(tmp_path)
    _, _, cohort = load_validation_data(tmp_path)
    save_graph(tmp_path, cohort, [("D", "M")])
    (tmp_path / "structure_validation_metadata.json").write_text(contents)
    with pytest.raises(ValueError, match="Invalid graph metadata. Rerun"):
        load_learned_parents(tmp_path, cohort)


@pytest.mark.parametrize("change", ["inputs", "raw_input", "edges"])
def test_graph_rejects_stale_metadata_and_modified_edges(tmp_path, change):
    make_cohort(tmp_path)
    _, _, cohort = load_validation_data(tmp_path)
    metadata = save_graph(tmp_path, cohort, [("D", "M")])
    if change == "inputs":
        metadata["inputs"]["train_fit"]["sha256"] = "old-training-hash"
    elif change == "raw_input":
        metadata["preprocessing"]["raw_input"]["sha256"] = "old-raw-hash"
    else:
        pd.DataFrame([("H", "M")], columns=["source", "target"]).to_csv(
            tmp_path / "learned_edges.csv", index=False,
        )
    (tmp_path / "structure_validation_metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="Stale learned graph"):
        load_learned_parents(tmp_path, cohort)


@pytest.mark.parametrize("edges", [
    [("D", "C"), ("C", "D")],
    [("M", "D")],
    [("UNKNOWN", "M")],
])
def test_invalid_graph_is_rejected_even_when_its_checksum_matches(tmp_path, edges):
    make_cohort(tmp_path)
    _, _, cohort = load_validation_data(tmp_path)
    save_graph(tmp_path, cohort, edges)
    with pytest.raises(ValueError, match="acyclic, use known variables"):
        load_learned_parents(tmp_path, cohort)


@pytest.mark.parametrize("parents", [[], ["H"], ["S", "D", "H"]])
def test_learned_parents_come_from_current_graph_and_are_sorted(tmp_path, parents):
    make_cohort(tmp_path)
    _, _, cohort = load_validation_data(tmp_path)
    # Other graph edges must not become parents of the outcome.
    save_graph(tmp_path, cohort, [("B", "C"), *[(parent, "M") for parent in parents]])
    assert load_learned_parents(tmp_path, cohort) == sorted(parents)
