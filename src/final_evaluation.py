"""Frozen five-model comparisons on random-game and chronological holdouts.

These evaluations refit the pre-specified models; they never select a model,
hyperparameter, graph, or threshold using either evaluation cohort.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from src.calibration import (
    LOGISTIC_MAX_ITER, N_BINS, calculate_calibration, get_predictions,
)
from src.chronological_evaluation import chronological_split, load_all_data
from src.ci_sensitivity import (
    BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, log_losses, paired_game_bootstrap,
)
from src.evaluate import evaluate_model
from src.provenance import file_sha256
from src.structure_learning import (
    ALPHA, FEATURES, HAND_PARENTS, MAX_INDEGREE, SCORING_METHOD,
    ConditionalMakeModel, learn_structure, prepare_data,
)
from src.validation import write_validation_metadata


DATA_DIR = Path("data/processed")
TABLE_DIR = Path("results/tables")
FIG_DIR = Path("results/figures")
MODEL_NAMES = (
    "League Average", "Logistic Regression", "Naive Bayes",
    "Hand-Designed BN", "Learned BN",
)


def frozen_protocol():
    """Fail if shared helpers drift from the declared final protocol."""
    if (ALPHA, HAND_PARENTS, LOGISTIC_MAX_ITER, N_BINS, SCORING_METHOD, MAX_INDEGREE) != (
        1.0, ["D", "T", "C", "H"], 1000, 10, "bic-d", 4,
    ):
        raise ValueError("Shared model settings differ from the frozen final protocol.")
    if (BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED) != (1000, 20261117):
        raise ValueError("Bootstrap settings differ from the frozen final protocol.")
    return {
        "primary_model": "Hand-Designed BN",
        "primary_parents": HAND_PARENTS,
        "primary_rationale": "Retain the original interpretable graph; no test-dependent selection.",
        "alpha": ALPHA,
        "features": FEATURES,
        "logistic_encoding": "one-hot; unknown categories ignored",
        "logistic_max_iter": LOGISTIC_MAX_ITER,
        "learned_structure": {
            "method": "hill climbing", "score": SCORING_METHOD,
            "max_indegree": MAX_INDEGREE,
            "forbidden_edges": [["M", feature] for feature in FEATURES],
            "fit_data": "training cohort only; relearned separately for each split",
        },
        "classification_threshold": 0.5,
        "calibration_bins": N_BINS,
        "calibration_binning": "equal-width; final bin includes 1",
        "model_selection_on_evaluation": False,
        "paired_comparisons": {
            "primary": "Hand-Designed BN",
            "comparators": [name for name in MODEL_NAMES if name != "Hand-Designed BN"],
            "replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED,
            "interval": "paired whole-game percentile 95%; retain all shots and game multiplicities",
            "estimand": "shot-weighted log loss(comparator) minus log loss(hand); positive favors hand",
            "refit_models_in_bootstrap": False,
        },
    }


def load_final_data(data_dir):
    """Verify exact preprocessing outputs and disjoint complete outer cohorts."""
    data_dir = Path(data_dir)
    report_path = data_dir / "preprocessing_report.json"
    report = json.loads(report_path.read_text())
    if report.get("schema_version") != 2:
        raise ValueError("Outdated preprocessing report; rerun preprocessing.")
    frames, inputs = [], {}
    required = ["RAW_ROW_ID", "GAME_ID", *FEATURES, "M"]
    for name in ("train", "test"):
        path = data_dir / f"{name}.csv"
        checksum = file_sha256(path)
        expected = report["splits"][name]
        if checksum != expected["sha256"]:
            raise ValueError(f"{path} hash does not match the preprocessing report.")
        frame = pd.read_csv(path)
        if not set(required).issubset(frame.columns):
            raise ValueError(f"{path} lacks required identifiers or model columns.")
        if frame.empty or frame[required].isna().any().any():
            raise ValueError(f"{path} is empty or has missing identifiers, features, or targets.")
        if not frame["RAW_ROW_ID"].is_unique or not frame["M"].isin([0, 1]).all():
            raise ValueError(f"{path} has duplicate shot identifiers or invalid targets.")
        ids = pd.to_numeric(frame["RAW_ROW_ID"], errors="coerce")
        if not (ids.notna() & ids.ge(0) & ids.lt(report["raw_rows"]) & ids.eq(ids.round())).all():
            raise ValueError(f"{path} has invalid raw row identifiers.")
        counts = {"shots": len(frame), "games": int(frame["GAME_ID"].nunique())}
        if any(counts[key] != expected[key] for key in counts):
            raise ValueError(f"{path} counts do not match the preprocessing report.")
        inputs[name] = {"file": str(path), "sha256": checksum, **counts}
        frames.append(frame)
    train, test = frames
    for column in ("RAW_ROW_ID", "GAME_ID"):
        if not set(train[column]).isdisjoint(test[column]):
            raise ValueError(f"Outer training/test cohorts overlap in {column}.")
    if len(train) + len(test) != report["clean_rows"]:
        raise ValueError("Outer cohorts do not cover the complete cleaned dataset.")
    cohort = {
        "inputs": inputs, "preprocessing_report_sha256": file_sha256(report_path),
        "preprocessing_report_file": str(report_path), "preprocessing": report,
    }
    return train, test, cohort


def cohort_summary(frame):
    summary = {
        "shots": len(frame), "games": int(frame["GAME_ID"].nunique()),
        "make_rate": float(frame["M"].astype(int).mean()),
    }
    if "date" in frame:
        summary.update({
            "date_min": str(frame["date"].min().date()),
            "date_max": str(frame["date"].max().date()),
            "dates": int(frame["date"].nunique()),
        })
    return summary


def paired_comparisons(per_shot):
    """Compare the declared primary to all four models on the same shot rows."""
    if not per_shot["RAW_ROW_ID"].is_unique or per_shot["GAME_ID"].isna().any():
        raise ValueError("Paired comparisons require unique shots and known whole-game IDs.")
    primary = "Hand-Designed BN"
    y = per_shot["actual"].to_numpy(dtype=int)
    primary_loss = log_losses(y, per_shot[primary])
    rows = []
    for comparator in MODEL_NAMES:
        if comparator == primary:
            continue
        comparator_loss = log_losses(y, per_shot[comparator])
        rows.append({
            "primary": primary, "comparator": comparator,
            "primary_log_loss": float(primary_loss.mean()),
            "comparator_log_loss": float(comparator_loss.mean()),
            **paired_game_bootstrap(
                comparator_loss - primary_loss, per_shot["GAME_ID"].to_numpy(),
                replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED,
            ),
        })
    return pd.DataFrame(rows)


def evaluate_cohort(prefix, train_raw, evaluation_raw, cohort, table_dir, fig_dir, data_dir):
    """Fit all models using one training frame; score identical held-out rows."""
    protocol = frozen_protocol()
    for column in ("RAW_ROW_ID", "GAME_ID"):
        if not set(train_raw[column]).isdisjoint(evaluation_raw[column]):
            raise ValueError(f"Training/evaluation overlap in {column}.")
    train, evaluation = prepare_data(train_raw), prepare_data(evaluation_raw)
    _, edges, parents = learn_structure(train)
    graph = nx.DiGraph()
    graph.add_nodes_from([*FEATURES, "M"])
    graph.add_edges_from(edges)
    if (set(graph) != {*FEATURES, "M"} or not nx.is_directed_acyclic_graph(graph)
            or graph.out_degree("M") or max(dict(graph.in_degree()).values()) > MAX_INDEGREE
            or sorted(graph.predecessors("M")) != sorted(parents)):
        raise ValueError("Learned graph violates the frozen structure protocol.")
    y_true, predictions = get_predictions(train, evaluation, parents)
    if tuple(predictions) != MODEL_NAMES:
        raise ValueError("The final comparison must include exactly the five frozen models.")
    metric_rows, calibration_tables = [], []
    per_shot = evaluation_raw[["RAW_ROW_ID", "GAME_ID"]].reset_index(drop=True).copy()
    per_shot["actual"] = np.asarray(y_true, dtype=int)
    for name, probabilities in predictions.items():
        probabilities = np.asarray(probabilities, dtype=float)
        if probabilities.shape != (len(evaluation),) or not np.isfinite(probabilities).all() \
                or not ((probabilities >= 0) & (probabilities <= 1)).all():
            raise ValueError(f"Invalid probabilities or evaluation cohort length for {name}.")
        per_shot[name] = probabilities
        table, ece = calculate_calibration(y_true, probabilities)
        table.insert(0, "model", name)
        calibration_tables.append(table)
        coverage = {"parents": "", "unseen_fraction": None, "observed_parent_configurations": None}
        if name in ("Hand-Designed BN", "Learned BN"):
            model_parents = HAND_PARENTS if name == "Hand-Designed BN" else parents
            cpt = ConditionalMakeModel(model_parents, ALPHA).fit(train)
            coverage = {
                "parents": ",".join(model_parents),
                "unseen_fraction": cpt.coverage(evaluation),
                "observed_parent_configurations": len(cpt.lookup) if model_parents else 1,
            }
        metric_rows.append({
            "model": name, "training_shots": len(train), "evaluation_shots": len(evaluation),
            "training_games": int(train_raw["GAME_ID"].nunique()),
            "evaluation_games": int(evaluation_raw["GAME_ID"].nunique()),
            **evaluate_model(name, y_true, probabilities), "ece": ece, **coverage,
        })
    metrics = pd.DataFrame(metric_rows).sort_values("log_loss")
    bins = pd.concat(calibration_tables, ignore_index=True)
    paired = paired_comparisons(per_shot)
    outputs = [
        table_dir / f"{prefix}_metrics.csv", table_dir / f"{prefix}_calibration_bins.csv",
        table_dir / f"{prefix}_learned_edges.csv", data_dir / f"{prefix}_predictions.csv",
        fig_dir / f"{prefix}_calibration.png",
        table_dir / f"{prefix}_paired_comparisons.csv",
    ]
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(outputs[0], index=False)
    bins.to_csv(outputs[1], index=False)
    pd.DataFrame(edges, columns=["source", "target"]).to_csv(outputs[2], index=False)
    per_shot.to_csv(outputs[3], index=False)
    paired.to_csv(outputs[5], index=False)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    for name, table in bins.groupby("model", sort=False):
        ax.plot(table["mean_predicted"], table["observed_rate"], marker="o", label=name)
    ax.set(xlabel="Mean predicted make probability", ylabel="Observed make rate",
           title="NBA Shot Quality: " + prefix.replace("_", " ").title(), xlim=(0, 1), ylim=(0, 1))
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outputs[4], dpi=300, bbox_inches="tight")
    plt.close(fig)
    details = {
        **cohort,
        "cohorts": {"training": cohort_summary(train_raw), "evaluation": cohort_summary(evaluation_raw)},
        "limitations": [
            "The outer test set was inspected by earlier repository scripts; this is a frozen-protocol refit, not a pristine new holdout.",
            "Chronology repartitions the same cleaned dataset and is a robustness check, not independent external validation.",
            "All five models see exactly the same evaluation shots; no evaluation outcomes enter fitting or structure learning.",
            "Paired percentile intervals condition on fitted models; they omit refitting/selection uncertainty and player dependence across games. Four nominal pairwise intervals are not simultaneous intervals.",
        ],
    }
    write_validation_metadata(
        table_dir / f"{prefix}_metadata.json", prefix, details,
        {**protocol, "learned_parents": parents}, outputs,
    )
    return metrics, per_shot


def main():
    train, test, cohort = load_final_data(DATA_DIR)
    # Parse chronology only from the already hash-verified frames.
    combined = load_all_data(train, test)
    dated_train, dated_test = combined.iloc[:len(train)].copy(), combined.iloc[len(train):].copy()
    final_metrics, _ = evaluate_cohort(
        "final_test", dated_train, dated_test, cohort, TABLE_DIR, FIG_DIR, DATA_DIR,
    )
    earlier, later = chronological_split(combined)
    original_test_ids = set(test["RAW_ROW_ID"])
    chronological_cohort = {
        **cohort,
        "chronological_split": {"unit": "unique date", "earlier_fraction": 0.8},
        "original_random_test_overlap": {
            "chronological_training_shots": int(earlier["RAW_ROW_ID"].isin(original_test_ids).sum()),
            "chronological_evaluation_shots": int(later["RAW_ROW_ID"].isin(original_test_ids).sum()),
        },
    }
    chronological_metrics, _ = evaluate_cohort(
        "chronological", earlier, later, chronological_cohort, TABLE_DIR, FIG_DIR, DATA_DIR,
    )
    print("\n=== FROZEN FINAL RANDOM-GAME COMPARISON ===")
    print(final_metrics.to_string(index=False))
    print("\n=== FROZEN CHRONOLOGICAL ROBUSTNESS COMPARISON ===")
    print(chronological_metrics.to_string(index=False))


if __name__ == "__main__":
    main()
