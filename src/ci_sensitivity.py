"""Matched validation models and paired game-cluster uncertainty intervals."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.graphs import HAND_PARENTS
from src.structure_learning import ConditionalMakeModel, prepare_data
from src.validation import load_learned_parents, load_validation_data, write_validation_metadata

DATA_DIR = Path('data/processed')
TABLE_DIR = Path('results/tables')
FIG_DIR = Path('results/figures')
BOOTSTRAP_REPLICATES = 1000
BOOTSTRAP_SEED = 20261117


def log_losses(target, probabilities):
    y = np.asarray(target)
    p = np.asarray(probabilities, dtype=float)
    if y.shape != p.shape or not np.isin(y, [0, 1]).all() or not np.isfinite(p).all():
        raise ValueError('Targets and finite probabilities must have matching shapes.')
    if ((p < 0) | (p > 1)).any():
        raise ValueError('Probabilities must lie in [0,1].')
    p = np.clip(p, 1e-10, 1 - 1e-10)
    return -(y * np.log(p) + (1 - y) * np.log1p(-p))


def paired_game_bootstrap(differences, games, replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED):
    """Resample games, retaining all their paired shot losses and multiplicities.

    The point estimate and each replicate weight shots equally. Intervals
    condition on the already fitted models; player overlap across games and
    model/structure-selection uncertainty are not captured.
    """
    differences = np.asarray(differences, dtype=float)
    games = np.asarray(games)
    if differences.ndim != 1 or len(differences) != len(games) or not len(games):
        raise ValueError('Paired losses and game IDs must be nonempty and aligned.')
    if not np.isfinite(differences).all() or pd.isna(games).any() or replicates < 1:
        raise ValueError('Finite losses, known game IDs, and positive replicates are required.')
    grouped = pd.DataFrame({'game': games, 'difference': differences}).groupby('game', sort=True)['difference'].agg(['sum', 'count'])
    if len(grouped) < 2:
        raise ValueError('At least two games are needed for clustered uncertainty.')
    rng = np.random.default_rng(seed)
    selections = rng.integers(0, len(grouped), size=(replicates, len(grouped)))
    estimates = grouped['sum'].to_numpy()[selections].sum(axis=1) / grouped['count'].to_numpy()[selections].sum(axis=1)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return {
        'log_loss_improvement': float(differences.mean()),
        'ci95_low': float(low), 'ci95_high': float(high),
        'games': len(grouped), 'shots': len(differences),
        'bootstrap_replicates': replicates, 'bootstrap_seed': seed,
    }


def comparisons(learned_parents):
    pairs = [
        ('Hand + clock', HAND_PARENTS, HAND_PARENTS + ['S']),
        ('Hand + dribbles', HAND_PARENTS, HAND_PARENTS + ['B']),
        ('Hand + clock and dribbles', HAND_PARENTS, HAND_PARENTS + ['S', 'B']),
    ]
    if 'H' not in learned_parents:
        pairs.append(('Learned + touch time', list(learned_parents), list(learned_parents) + ['H']))
    return pairs


def main():
    train_raw, validation_raw, cohort = load_validation_data(DATA_DIR)
    parents = load_learned_parents(TABLE_DIR, cohort)
    train, validation = prepare_data(train_raw), prepare_data(validation_raw)
    y = validation['M'].astype(int).to_numpy()
    predictions = validation_raw[['RAW_ROW_ID', 'GAME_ID']].reset_index(drop=True).copy()
    predictions['actual'] = y
    cache = {}
    rows = []
    for name, reduced, expanded in comparisons(parents):
        for key, model_parents in [('reduced', reduced), ('expanded', expanded)]:
            state = tuple(sorted(model_parents))
            if state not in cache:
                model = ConditionalMakeModel(model_parents, alpha=1.0).fit(train)
                p = model.predict_proba(validation)
                cache[state] = (p, model.coverage(validation))
            p, _ = cache[state]
            predictions[f'{name}: {key}'] = p
        p_reduced, coverage_reduced = cache[tuple(sorted(reduced))]
        p_expanded, coverage_expanded = cache[tuple(sorted(expanded))]
        loss_reduced, loss_expanded = log_losses(y, p_reduced), log_losses(y, p_expanded)
        rows.append({
            'comparison': name, 'reduced_parents': ','.join(reduced), 'expanded_parents': ','.join(expanded),
            'reduced_log_loss': float(loss_reduced.mean()), 'expanded_log_loss': float(loss_expanded.mean()),
            'reduced_unseen_fraction': coverage_reduced, 'expanded_unseen_fraction': coverage_expanded,
            **paired_game_bootstrap(loss_reduced - loss_expanded, validation_raw['GAME_ID']),
        })
    results = pd.DataFrame(rows)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    table = TABLE_DIR / 'ci_predictive_sensitivity.csv'
    details = DATA_DIR / 'ci_validation_predictions.csv'
    figure = FIG_DIR / 'ci_predictive_sensitivity.png'
    results.to_csv(table, index=False)
    predictions.to_csv(details, index=False)
    fig, ax = plt.subplots(figsize=(8, 4.5), layout='constrained')
    positions = np.arange(len(results))
    # Percentile intervals need not contain the sample mean; draw endpoints directly.
    ax.hlines(positions, results.ci95_low, results.ci95_high, color='#24738a', linewidth=2)
    ax.scatter(results.log_loss_improvement, positions, color='#153c52', zorder=3)
    ax.axvline(0, color='black', linewidth=1, linestyle='--')
    ax.set_yticks(positions, results.comparison)
    ax.invert_yaxis()
    ax.set_xlabel('Validation log-loss improvement (positive favors added features)')
    ax.set_title('Predictive sensitivity: paired game-cluster 95% intervals')
    ax.grid(axis='x', alpha=0.2)
    fig.savefig(figure, dpi=250, bbox_inches='tight')
    plt.close(fig)
    write_validation_metadata(
        TABLE_DIR / 'ci_predictive_sensitivity_metadata.json', 'ci_predictive_sensitivity', cohort,
        {
            'alpha': 1.0, 'split': 'validation', 'fit_split': 'train_fit',
            'comparisons': [{'name': name, 'reduced': reduced, 'expanded': expanded} for name, reduced, expanded in comparisons(parents)],
            'replicates': BOOTSTRAP_REPLICATES, 'seed': BOOTSTRAP_SEED,
            'interval': 'paired game-cluster percentile 95%; resample games with replacement, keep all shots',
            'estimand': 'shot-weighted reduced log loss minus expanded log loss; positive favors expanded model',
            'limitations': 'Exploratory after validation-based selection; conditional on fitted models. No formal CI test; cross-game player dependence and refitting uncertainty omitted.',
            'detailed_predictions': str(details),
        }, [table, figure, details],
        [TABLE_DIR / 'learned_edges.csv', TABLE_DIR / 'structure_validation_metadata.json'],
    )
    print(results.to_string(index=False))


if __name__ == '__main__':
    main()
