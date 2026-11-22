"""Shared inputs and artifact provenance for the validation experiments."""

from datetime import datetime, timezone
import json
from pathlib import Path

import networkx as nx
import pandas as pd

from src.preprocess import FEATURES
from src.provenance import collect_provenance, file_sha256


def load_validation_data(data_dir):
    """Reject stale or overlapping splits before any models are fitted."""
    data_dir = Path(data_dir)
    report_path = data_dir / "preprocessing_report.json"
    if not report_path.exists():
        raise FileNotFoundError("Missing preprocessing report. Run python -m src.preprocess first.")
    report = json.loads(report_path.read_text())
    if report.get("schema_version") != 2:
        raise ValueError("Outdated preprocessing report. Run python -m src.preprocess first.")

    frames = []
    inputs = {}
    for name in ("train_fit", "validation"):
        path = data_dir / f"{name}.csv"
        checksum = file_sha256(path)
        expected = report["splits"][name]
        if checksum != expected["sha256"]:
            raise ValueError(f"{path} does not match the preprocessing report; rerun preprocessing.")
        frame = pd.read_csv(path)
        required = ["RAW_ROW_ID", "GAME_ID", *FEATURES, "M"]
        if not set(required).issubset(frame.columns):
            raise ValueError(f"{path} lacks required identifiers or model columns.")
        if frame.empty or frame[["RAW_ROW_ID", "GAME_ID", "M"]].isna().any().any():
            raise ValueError(f"{path} has no shots or has missing identifiers/targets.")
        if not frame["RAW_ROW_ID"].is_unique or not frame["M"].isin([0, 1]).all():
            raise ValueError(f"{path} has duplicate shot identifiers or invalid targets.")
        counts = {"shots": len(frame), "games": int(frame["GAME_ID"].nunique())}
        if any(counts[key] != expected[key] for key in counts):
            raise ValueError(f"{path} counts do not match the preprocessing report.")
        inputs[name] = {"file": path.name, "sha256": checksum, **counts}
        frames.append(frame)

    train, validation = frames
    for column in ("RAW_ROW_ID", "GAME_ID"):
        if not set(train[column]).isdisjoint(validation[column]):
            raise ValueError(f"Training-fit and validation overlap in {column}.")
    cohort = {
        "inputs": inputs,
        "preprocessing_report_sha256": file_sha256(report_path),
        "preprocessing": report,
    }
    return train, validation, cohort


def write_validation_metadata(path, experiment, cohort, configuration, outputs, dependencies=()):
    """Keep a portable copy of preprocessing metadata beside committed results."""
    report = {
        "schema_version": 1,
        "experiment": experiment,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        **cohort,
        "configuration": configuration,
        "provenance": collect_provenance(),
        "dependencies": {
            Path(item).name: {"file": str(item), "sha256": file_sha256(item)}
            for item in dependencies
        },
        "outputs": {
            Path(item).name: {"file": str(item), "sha256": file_sha256(item)}
            for item in outputs
        },
    }
    Path(path).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def load_learned_parents(results_dir, cohort):
    """Use only a verified structure learned on the current validation inputs."""
    results_dir = Path(results_dir)
    edges_path = results_dir / "learned_edges.csv"
    metadata_path = results_dir / "structure_validation_metadata.json"
    if not edges_path.exists() or not metadata_path.exists():
        raise FileNotFoundError("Missing graph/provenance. Run python -m src.structure_learning first.")
    try:
        metadata = json.loads(metadata_path.read_text())
        matches = (
            metadata["schema_version"] == 1
            and metadata["experiment"] == "structure_validation"
            and metadata["inputs"] == cohort["inputs"]
            and metadata["preprocessing"]["raw_input"]["sha256"]
            == cohort["preprocessing"]["raw_input"]["sha256"]
            and metadata["outputs"]["learned_edges.csv"]["sha256"] == file_sha256(edges_path)
        )
    except (ValueError, KeyError, TypeError) as error:
        raise ValueError("Invalid graph metadata. Rerun python -m src.structure_learning.") from error
    if not matches:
        raise ValueError("Stale learned graph or changed inputs. Rerun python -m src.structure_learning.")

    edges = pd.read_csv(edges_path)
    if set(edges.columns) != {"source", "target"}:
        raise ValueError("Learned graph must have source and target columns.")
    graph = nx.DiGraph()
    graph.add_nodes_from([*FEATURES, "M"])
    graph.add_edges_from(edges[["source", "target"]].itertuples(index=False, name=None))
    if (
        set(graph) != {*FEATURES, "M"}
        or not nx.is_directed_acyclic_graph(graph)
        or graph.out_degree("M") != 0
    ):
        raise ValueError("Learned graph must be acyclic, use known variables, and have no edges out of M.")
    return sorted(graph.predecessors("M"))
