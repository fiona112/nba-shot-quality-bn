"""Training-only graph, CPT, inference, runtime, and stability summaries.

These are descriptive analyses. Edges do not establish causal effects, empirical
conditional rates need not equal model marginals, and three subsamples provide
an exploratory sensitivity check rather than a confidence interval.
"""

import inspect
import json
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from pgmpy.estimators import HillClimbSearch
from scipy.stats import beta

from src.graphs import FEATURES, HAND_EDGES, HAND_PARENT_MAP, VARIABLES
from src.structure_learning import (
    ALPHA, MAX_INDEGREE, SCORING_METHOD, learn_structure, prepare_data,
)
from src.validation import (
    load_learned_parents, load_validation_data, write_validation_metadata,
)
from src.variable_elimination import BayesianNetwork


DATA_DIR = Path("data/processed")
TABLE_DIR = Path("results/tables")
FIG_DIR = Path("results/figures")
STABILITY_SEEDS = (13, 37, 73)
STABILITY_GAME_FRACTION = 0.8
QUERY_SPECIFICATIONS = (
    ("population", {}),
    ("close_two_tight_defender", {"D": "<5", "T": "2", "C": "0-2"}),
    ("close_two_open", {"D": "<5", "T": "2", "C": "6+"}),
    ("long_three_moderate_defender", {"D": "25-30", "T": "3", "C": "4-6"}),
)
SYMBOL_LEGEND = {
    "T": "shot value (2/3 points)", "D": "shot distance (feet)",
    "C": "closest defender distance (feet)", "S": "effective shot clock (seconds)",
    "B": "dribbles", "H": "touch time (seconds)", "M": "made shot (0/1)",
}


def summarize_make_cpt(train, model):
    """Observed hand-CPT strata, with counts and marginal Beta posterior limits.

    A Beta(alpha, alpha) prior for a binary CPT row is equivalent to the
    Laplace estimate. These intervals use the row's shot likelihood and do not
    adjust for dependence between shots from the same game.
    """
    parents = list(HAND_PARENT_MAP["M"])
    work = train[parents].astype("string").copy()
    work["made"] = train["M"].astype(str).eq("1").astype(int)
    counts = work.groupby(parents, observed=True, dropna=True)["made"].agg(
        training_shots="count", made="sum",
    ).reset_index()
    counts["missed"] = counts["training_shots"] - counts["made"]
    counts["empirical_make_rate"] = counts["made"] / counts["training_shots"]
    cpt = model.factors["M"].table
    cpt = cpt.loc[cpt["M"].eq("1"), parents + ["p"]].rename(
        columns={"p": "laplace_make_probability"},
    )
    result = counts.merge(cpt, on=parents, validate="one_to_one")
    result["posterior_ci95_lower"] = beta.ppf(
        0.025, result["made"] + model.alpha, result["missed"] + model.alpha,
    )
    result["posterior_ci95_upper"] = beta.ppf(
        0.975, result["made"] + model.alpha, result["missed"] + model.alpha,
    )
    return result.sort_values(
        ["training_shots", *parents], ascending=[False, *([True] * len(parents))],
    ).reset_index(drop=True)


def run_inference_queries(train, model, specifications=QUERY_SPECIFICATIONS):
    """Evaluate fixed partial-evidence examples and report training support."""
    rows, timings = [], []
    for query_id, evidence in specifications:
        mask = pd.Series(True, index=train.index)
        for variable, value in evidence.items():
            mask &= train[variable].astype(str).eq(str(value))
        supported = train.loc[mask]
        if supported.empty:
            raise ValueError(f"Inference query {query_id} has no training support: {evidence}")
        start = perf_counter()
        distribution = model.query("M", evidence)
        elapsed = perf_counter() - start
        rows.append({
            "query": query_id,
            "evidence": json.dumps(evidence, sort_keys=True),
            "training_shots": len(supported),
            "training_games": int(supported["GAME_ID"].nunique()),
            "empirical_make_rate": float(supported["M"].astype(str).eq("1").mean()),
            "ve_make_probability": distribution["1"],
        })
        timings.append({"operation": f"query:{query_id}", "seconds": elapsed, "calls": 1})
    return pd.DataFrame(rows), pd.DataFrame(timings)


def _graph_positions():
    """Identical ring layouts make graph comparison easy and avoid hidden nodes.

    A straight edge between nodes on this convex boundary cannot pass through
    another node. This avoids an apparent T -> C arrow when D -> C crosses T
    in a layered layout, for example.
    """
    ring = ["D", "S", "B", "H", "M", "C", "T"]
    angles = np.deg2rad(100 - np.arange(len(ring)) * 360 / len(ring))
    return {node: (np.cos(angle), np.sin(angle)) for node, angle in zip(ring, angles)}


def plot_graphs(learned_edges, path):
    """Display the actual hand graph and verified train-fit learned graph."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 7))
    for ax, title, edges in zip(
        axes, ("Hand-designed BN", "Learned BN (training-fit data)"),
        (HAND_EDGES, learned_edges),
    ):
        graph = nx.DiGraph()
        graph.add_nodes_from(VARIABLES)
        graph.add_edges_from(edges)
        if not nx.is_directed_acyclic_graph(graph):
            raise ValueError("Graph figure requires a DAG.")
        nx.draw_networkx(
            graph, pos=_graph_positions(), ax=ax,
            node_color=["#1d3557" if n == "M" else "#187d8d" for n in graph],
            node_size=1150, font_size=15, font_color="white", font_weight="bold",
            edge_color="#4c5666", width=1.6, arrows=True, arrowsize=20,
            arrowstyle="-|>", connectionstyle="arc3,rad=0.0",
            min_source_margin=18, min_target_margin=18,
        )
        ax.set_title(f"{title}\n{len(edges)} directed edges", fontsize=14, pad=18)
        ax.set_aspect("equal")
        ax.margins(0.25)
        ax.set_axis_off()
    fig.suptitle("Shot-quality Bayesian networks", fontsize=18, y=0.98)
    legend = "    ".join(f"{v}: {SYMBOL_LEGEND[v]}" for v in ("T", "D", "C"))
    legend += "\n" + "    ".join(f"{v}: {SYMBOL_LEGEND[v]}" for v in ("S", "B", "H", "M"))
    fig.text(0.5, 0.09, legend, ha="center", va="center", fontsize=10, linespacing=1.8)
    fig.text(
        0.5, 0.025,
        "Arrows specify a probabilistic factorization; observational data do not establish causal effects.",
        ha="center", fontsize=10, color="#4c5666",
    )
    fig.subplots_adjust(bottom=0.22, top=0.86, wspace=0.3)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def sample_training_games(train, seed, fraction=STABILITY_GAME_FRACTION):
    """Subsample complete training games, preserving within-game shots."""
    if not 0 < fraction <= 1:
        raise ValueError("Game subsample fraction must lie in (0, 1].")
    games = np.sort(train["GAME_ID"].unique())
    count = max(1, int(np.ceil(fraction * len(games))))
    selected = np.random.default_rng(seed).choice(games, size=count, replace=False)
    return train.loc[train["GAME_ID"].isin(selected)].reset_index(drop=True)


def structure_stability(train_raw, full_edges, seeds=STABILITY_SEEDS,
                        fraction=STABILITY_GAME_FRACTION):
    """Learn on three fixed training-game subsamples with unchanged constraints."""
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Stability seeds must be nonempty and distinct.")
    selected_edges, rows = [], []
    for seed in seeds:
        subset = sample_training_games(train_raw, seed, fraction)
        start = perf_counter()
        _, edges, parents = learn_structure(prepare_data(subset))
        elapsed = perf_counter() - start
        selected_edges.append(set(edges))
        rows.append({
            "seed": seed, "game_fraction": fraction,
            "training_games": int(subset["GAME_ID"].nunique()),
            "training_shots": len(subset), "edge_count": len(edges),
            "make_parents": ",".join(parents), "learning_seconds": elapsed,
        })
    full = set(map(tuple, full_edges))
    hand = set(map(tuple, HAND_EDGES))
    candidates = sorted(full | hand | set().union(*selected_edges))
    frequencies = pd.DataFrame([
        {
            "source": source, "target": target,
            "subsamples_selected": sum((source, target) in edges for edges in selected_edges),
            "subsamples_total": len(seeds),
            "frequency": sum((source, target) in edges for edges in selected_edges) / len(seeds),
            "full_train_fit_edge": (source, target) in full,
            "hand_edge": (source, target) in hand,
        }
        for source, target in candidates
    ])
    return frequencies, pd.DataFrame(rows)


def main():
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    train_raw, _, cohort = load_validation_data(DATA_DIR)
    # This verifies both the learned-edge checksum and its training cohort.
    learned_make_parents = load_learned_parents(TABLE_DIR, cohort)
    edges_path = TABLE_DIR / "learned_edges.csv"
    learned_edges = list(pd.read_csv(edges_path).itertuples(index=False, name=None))
    train = prepare_data(train_raw)
    train["GAME_ID"] = train_raw["GAME_ID"].to_numpy()
    start = perf_counter()
    model = BayesianNetwork(alpha=ALPHA).fit(train)
    fit_seconds = perf_counter() - start
    cpt = summarize_make_cpt(train, model)
    queries, timings = run_inference_queries(train, model)
    timings = pd.concat([
        pd.DataFrame([{"operation": "fit:hand_full_bn", "seconds": fit_seconds, "calls": 1}]),
        timings,
    ], ignore_index=True)
    frequencies, runs = structure_stability(train_raw, learned_edges)
    outputs = []
    for name, table in (
        ("cpt_summary", cpt), ("inference_queries", queries), ("inference_runtime", timings),
        ("structure_stability_edges", frequencies), ("structure_stability_runs", runs),
    ):
        path = TABLE_DIR / f"{name}.csv"
        table.to_csv(path, index=False)
        outputs.append(path)
    figure_path = FIG_DIR / "bayesian_network_graphs.png"
    plot_graphs(learned_edges, figure_path)
    outputs.append(figure_path)
    signature = inspect.signature(HillClimbSearch.estimate)
    write_validation_metadata(
        TABLE_DIR / "interpretation_metadata.json", "training_interpretation", cohort,
        {
            "data_used": "train_fit only; validation loaded solely to verify cohort provenance",
            "alpha": ALPHA, "hand_parent_map": HAND_PARENT_MAP,
            "verified_learned_make_parents": learned_make_parents,
            "queries": [{"query": name, "evidence": evidence} for name, evidence in QUERY_SPECIFICATIONS],
            "cpt_interval": {
                "prior": "independent Beta(alpha, alpha) for each binary make-CPT row",
                "level": 0.95, "method": "equal-tailed posterior quantiles",
                "scope": "observed training strata only; no adjustment for within-game dependence",
            },
            "stability": {
                "seeds": list(STABILITY_SEEDS), "training_game_fraction": STABILITY_GAME_FRACTION,
                "sampling": "without replacement, complete games, independently for each fixed seed",
                "scoring_method": SCORING_METHOD, "max_indegree": MAX_INDEGREE,
                "forbidden_edges": [["M", feature] for feature in FEATURES],
                "search_stopping_defaults": {
                    key: signature.parameters[key].default
                    for key in ("epsilon", "max_iter", "tabu_length")
                },
                "caveat": "Greedy search can stop at a local optimum; stopping/convergence diagnostics are not exposed by this wrapper. Three overlapping subsamples give descriptive sensitivity frequencies, not confidence intervals or causal orientation evidence.",
                "edge_table_scope": "union of hand, full-training learned, and subsample-selected directed edges",
            },
            "runtime": "wall-clock seconds for one fit and one invocation of each query; hardware-dependent, no benchmark averaging",
            "interpretation": "Empirical evidence-stratum rates and BN marginal predictions both target P(M=1|evidence), but may differ because the BN imposes conditional-independence assumptions and uses smoothing. All comparisons are associational.",
        },
        outputs, dependencies=[edges_path, TABLE_DIR / "structure_validation_metadata.json"],
    )
    print("Saved training-only graph, CPT, inference, runtime, and stability summaries.")
    print(queries.to_string(index=False))
    print(runs.to_string(index=False))


if __name__ == "__main__":
    main()
