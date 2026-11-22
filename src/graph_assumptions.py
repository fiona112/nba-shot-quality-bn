"""Verify selected conditional-independence restrictions using d-separation.

These checks establish what each graph assumes. They do not test whether the
NBA observations satisfy those assumptions; empirical diagnostics are separate.
Neither a learned edge nor a d-separation claim establishes a causal effect.
"""

from pathlib import Path

import networkx as nx
import pandas as pd

from src.graphs import FEATURES, HAND_EDGES, HAND_PARENTS, VARIABLES
from src.validation import (
    load_learned_parents,
    load_validation_data,
    write_validation_metadata,
)


DATA_DIR = Path("data/processed")
RESULTS_DIR = Path("results/tables")


def build_claims(learned_parents):
    """Return predeclared questions using the actual learned make-parent set.

    M is constrained to be a sink in the learned graph. Its local Markov
    restriction therefore separates M from every omitted feature given all
    its parents. When H is a parent, no learned touch-time claim is emitted.
    """
    learned_parents = list(learned_parents)
    if len(set(learned_parents)) != len(learned_parents) or not set(learned_parents) <= set(FEATURES):
        raise ValueError("Learned make parents must be distinct known features.")

    def claim(identifier, model, x, y, given):
        return {"id": identifier, "model": model, "x": list(x), "y": list(y), "given": list(given)}

    claims = [
        claim("hand_clock", "Hand BN", ["M"], ["S"], HAND_PARENTS),
        claim("hand_dribbles", "Hand BN", ["M"], ["B"], HAND_PARENTS),
        claim("hand_joint", "Hand BN", ["M"], ["S", "B"], HAND_PARENTS),
        claim("hand_roots", "Hand BN", ["S"], ["D"], []),
        claim("nb_distance_type", "Naive Bayes", ["D"], ["T"], ["M"]),
        claim("nb_dribbles_touch", "Naive Bayes", ["B"], ["H"], ["M"]),
    ]
    omitted = [feature for feature in FEATURES if feature not in learned_parents]
    if "H" in omitted:
        claims.append(claim("learned_touch", "Learned BN", ["M"], ["H"], learned_parents))
    if omitted:
        claims.append(claim("learned_joint", "Learned BN", ["M"], omitted, learned_parents))
    return claims


def graph_for_model(model, learned_edges):
    """Build the full graph for a model, retaining nodes without any edges."""
    if model == "Hand BN":
        edges = HAND_EDGES
    elif model == "Naive Bayes":
        edges = [("M", feature) for feature in FEATURES]
    elif model == "Learned BN":
        edges = list(learned_edges)
    else:
        raise ValueError(f"Unknown graph model: {model}")
    graph = nx.DiGraph()
    graph.add_nodes_from(VARIABLES)
    graph.add_edges_from(edges)
    if set(graph) != set(VARIABLES) or not nx.is_directed_acyclic_graph(graph):
        raise ValueError("Graph must be acyclic and use only project variables.")
    if model == "Learned BN" and graph.out_degree("M"):
        raise ValueError("The learned shot-outcome graph must have no edges out of M.")
    return graph


def verify_claims(learned_parents, learned_edges):
    """Verify every requested model restriction and fail on a graph mismatch."""
    learned = graph_for_model("Learned BN", learned_edges)
    if set(learned.predecessors("M")) != set(learned_parents):
        raise ValueError("Learned parent list does not match the learned graph.")
    rows = []
    for claim in build_claims(learned_parents):
        graph = graph_for_model(claim["model"], learned_edges)
        separated = bool(nx.is_d_separator(graph, set(claim["x"]), set(claim["y"]), set(claim["given"])))
        if not separated:
            raise ValueError(f"Claim {claim['id']} is not implied by the actual graph.")
        rows.append({
            "id": claim["id"],
            "model": claim["model"],
            "x": ",".join(claim["x"]),
            "y": ",".join(claim["y"]),
            "given": ",".join(claim["given"]),
            "d_separated": separated,
            "interpretation": "Graph-implied model restriction; not evidence of independence in NBA data.",
        })
    return rows


def main():
    _, _, cohort = load_validation_data(DATA_DIR)
    learned_parents = load_learned_parents(RESULTS_DIR, cohort)
    edges_path = RESULTS_DIR / "learned_edges.csv"
    learned_edges = list(pd.read_csv(edges_path).itertuples(index=False, name=None))
    rows = verify_claims(learned_parents, learned_edges)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "graph_assumptions.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    write_validation_metadata(
        RESULTS_DIR / "graph_assumptions_metadata.json",
        "graph_assumptions",
        cohort,
        {
            "method": "networkx.is_d_separator",
            "learned_make_parents": learned_parents,
            "claims": build_claims(learned_parents),
            "interpretation": "D-separation checks graph-implied assumptions only; no observed-data test is performed.",
        },
        [output_path],
        dependencies=[edges_path, RESULTS_DIR / "structure_validation_metadata.json"],
    )
    print(pd.DataFrame(rows).drop(columns="interpretation").to_string(index=False))
    print("D-separation verifies graph restrictions, not independence in the observed NBA data.")


if __name__ == "__main__":
    main()
