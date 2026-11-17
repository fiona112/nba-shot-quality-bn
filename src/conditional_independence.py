"""Descriptive conditional mutual information with explicit support diagnostics."""

from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from src.graph_assumptions import build_claims, graph_for_model
from src.validation import load_learned_parents, load_validation_data, write_validation_metadata

DATA_DIR = Path('data/processed')
TABLE_DIR = Path('results/tables')


def conditional_mutual_information(data, x, y, given=()):
    """Plug-in I(X;Y|Z), in nats, on complete cases for the selected variables.

    X and Y may each be a tuple of variables. Expected counts describe support,
    not a claim that shots are independent sampling units or a valid chi-square
    reference distribution. No significance p-value is computed.
    """
    x, y, given = list(x), list(y), list(given)
    required = x + y + given
    if not x or not y or len(set(required)) != len(required):
        raise ValueError('X and Y must be nonempty; variable groups must be disjoint.')
    frame = data[required].dropna()
    if frame.empty:
        raise ValueError('No complete observations for this claim.')
    codes = {}
    for name, columns in [('x', x), ('y', y), ('z', given)]:
        codes[name] = (
            pd.factorize(pd.MultiIndex.from_frame(frame[columns]))[0]
            if columns else np.zeros(len(frame), dtype=int)
        )
    encoded = pd.DataFrame(codes)
    global_cells = encoded.x.nunique() * encoded.y.nunique()
    stats = {
        'input_shots': len(data), 'shots': len(frame),
        'dropped_missing_shots': len(data) - len(frame),
        'x_states': int(encoded.x.nunique()), 'y_states': int(encoded.y.nunique()),
        'conditioning_strata': 0, 'informative_strata': 0,
        'effective_table_cells': 0, 'empty_joint_cells': 0,
        'global_grid_zero_expected_cells': 0,
        'expected_cells_below_5': 0, 'expected_cells_below_1': 0,
        'degrees_of_freedom_descriptive': 0,
    }
    information = 0.0
    sparse_shots = constant_shots = 0
    for _, group in encoded.groupby('z', sort=True):
        counts = pd.crosstab(group.x, group.y).to_numpy(dtype=float)
        rows, columns = counts.shape
        n = len(group)
        expected = np.outer(counts.sum(axis=1), counts.sum(axis=0)) / n
        occupied = counts > 0
        information += float((counts[occupied] * np.log(counts[occupied] / expected[occupied])).sum())
        stats['conditioning_strata'] += 1
        stats['effective_table_cells'] += int(counts.size)
        stats['empty_joint_cells'] += int((counts == 0).sum())
        stats['global_grid_zero_expected_cells'] += int(global_cells - counts.size)
        stats['expected_cells_below_5'] += int((expected < 5).sum())
        stats['expected_cells_below_1'] += int((expected < 1).sum())
        stats['degrees_of_freedom_descriptive'] += (rows - 1) * (columns - 1)
        if rows > 1 and columns > 1:
            stats['informative_strata'] += 1
            if (expected < 5).any():
                sparse_shots += n
        else:
            constant_shots += n
    cmi = max(0.0, information / len(frame))
    return {
        **stats, 'cmi_nats': cmi, 'g2_descriptive': 2 * len(frame) * cmi,
        'shots_in_sparse_informative_strata_fraction': sparse_shots / len(frame),
        'shots_in_constant_strata_fraction': constant_shots / len(frame),
    }


def main():
    train, _, cohort = load_validation_data(DATA_DIR)
    parents = load_learned_parents(TABLE_DIR, cohort)
    edges = list(pd.read_csv(TABLE_DIR / 'learned_edges.csv').itertuples(index=False, name=None))
    rows = []
    for claim in build_claims(parents):
        graph = graph_for_model(claim['model'], edges)
        implied = nx.is_d_separator(graph, set(claim['x']), set(claim['y']), set(claim['given']))
        rows.append({
            'claim_id': claim['id'], 'model': claim['model'],
            'x': ','.join(claim['x']), 'y': ','.join(claim['y']),
            'given': ','.join(claim['given']), 'graph_implied': bool(implied),
            'split': 'train_fit',
            **conditional_mutual_information(train, claim['x'], claim['y'], claim['given']),
        })
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    output = TABLE_DIR / 'conditional_independence.csv'
    pd.DataFrame(rows).to_csv(output, index=False)
    write_validation_metadata(
        TABLE_DIR / 'conditional_independence_metadata.json',
        'conditional_independence', cohort,
        {
            'split': 'train_fit', 'estimator': 'unsmoothed plug-in conditional mutual information',
            'units': 'nats', 'missing_policy': 'complete cases per claim',
            'interpretation': 'descriptive effect sizes, not cluster-valid hypothesis tests',
            'bias_warning': 'finite-sample CMI is upward biased, especially in sparse strata',
            'support_grid': 'effective tables use states observed within each stratum; global-grid zero expected cells include absent marginals',
            'claims': build_claims(parents),
        }, [output],
        [TABLE_DIR / 'learned_edges.csv', TABLE_DIR / 'structure_validation_metadata.json'],
    )
    print(pd.DataFrame(rows)[['claim_id', 'graph_implied', 'cmi_nats', 'shots_in_sparse_informative_strata_fraction']].to_string(index=False))


if __name__ == '__main__':
    main()
