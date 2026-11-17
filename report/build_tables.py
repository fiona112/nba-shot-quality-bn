"""Generate report numbers and compact tables from committed experiment outputs."""

from pathlib import Path
import json
import hashlib

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / 'results' / 'tables'
OUT = Path(__file__).resolve().parent / 'generated'
DISPLAY = {'Hand-Designed BN': 'Hand BN', 'Logistic Regression': 'Logistic regression', 'Naive Bayes': 'Naive Bayes', 'Learned BN': 'Learned BN', 'League Average': 'League average'}
ORDER = ['Hand-Designed BN', 'Learned BN', 'Logistic Regression', 'Naive Bayes', 'League Average']
METADATA = ['calibration', 'ablation_validation', 'structure_validation',
            'graph_assumptions', 'conditional_independence',
            'ci_predictive_sensitivity', 'final_test', 'chronological',
            'interpretation', 'preprocessing_sensitivity', 'latent_em',
            'player_comparison']


def read(name):
    return pd.read_csv(TABLES / f'{name}.csv')


def table(name, header, rows, columns):
    lines = [r'\begin{tabular}{' + columns + '}', r'\toprule', ' & '.join(header) + r'\\', r'\midrule']
    lines.extend(' & '.join(map(str, row)) + r'\\' for row in rows)
    lines.extend([r'\bottomrule', r'\end{tabular}'])
    (OUT / f'{name}.tex').write_text('\n'.join(lines) + '\n')


def main():
    OUT.mkdir(exist_ok=True)
    # Source tables and committed figures must still match their recorded runs.
    inputs = {}
    for name in METADATA:
        metadata_path = TABLES / f'{name}_metadata.json'
        inputs[str(metadata_path.relative_to(ROOT))] = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
        metadata = json.loads(metadata_path.read_text())
        for spec in metadata['outputs'].values():
            path = Path(spec['file'])
            if path.parts[:2] == ('data', 'processed'):
                continue  # Detailed predictions are reproducible local artifacts.
            path = ROOT / path
            if hashlib.sha256(path.read_bytes()).hexdigest() != spec['sha256']:
                raise ValueError(f'Report input no longer matches run metadata: {path}')
            inputs[str(path.relative_to(ROOT))] = spec['sha256']
    val = read('calibration_metrics').set_index('model')
    test = read('final_test_metrics').set_index('model')
    chrono = read('chronological_metrics').set_index('model')
    em = read('latent_em_test_metrics').iloc[0]
    em_validation = read('latent_em_validation').set_index('k').loc[int(em['k'])]
    rows = []
    for model in ORDER:
        rows.append([DISPLAY[model], f'{val.loc[model, "log_loss"]:.4f}', f'{test.loc[model, "log_loss"]:.4f}', f'{test.loc[model, "brier_score"]:.4f}', f'{test.loc[model, "accuracy"]:.4f}', f'{test.loc[model, "ece"]:.4f}'])
    rows.insert(1, [f'Latent BN ($K={int(em["k"])}$)', f'{em_validation["val_log_loss"]:.4f}', *[f'{em[key]:.4f}' for key in ['log_loss', 'brier_score', 'accuracy', 'ece']]])
    table('main_results', ['Model', 'Val. LL', 'Test LL', 'Test Brier', 'Test Acc.', 'Test ECE'], rows, 'lrrrrr')
    table('chronology', ['Model', 'Log loss', 'Brier', 'Accuracy', 'ECE'], [[DISPLAY[m], *[f'{chrono.loc[m,k]:.4f}' for k in ['log_loss','brier_score','accuracy','ece']]] for m in ORDER], 'lrrrr')
    claims = read('conditional_independence').set_index('claim_id')
    labels = {
        'hand_clock': r'$M\perp S\mid D,T,C,H$',
        'hand_dribbles': r'$M\perp B\mid D,T,C,H$',
        'hand_joint': r'$M\perp(S,B)\mid D,T,C,H$',
        'hand_roots': r'$S\perp D$',
        'nb_distance_type': r'$D\perp T\mid M$',
        'nb_dribbles_touch': r'$B\perp H\mid M$',
        'learned_touch': r'$M\perp H\mid D,C$',
        'learned_joint': r'$M\perp(T,S,B,H)\mid D,C$',
    }
    table('ci', ['Graph restriction', 'CMI (nats)', 'Sparse shots'], [[label, f'{claims.loc[key,"cmi_nats"]:.5f}', f'{100*claims.loc[key,"shots_in_sparse_informative_strata_fraction"]:.1f}\\%'] for key,label in labels.items() if key in claims.index], 'lrr')
    sensitivity = read('ci_predictive_sensitivity')
    table('sensitivity', ['Added parents', r'$\Delta$ log loss', '95\\% game interval'], [[r.comparison.replace('Hand + ', 'Hand: ').replace('Learned + ', 'Learned: '), f'{r.log_loss_improvement:+.5f}', f'[{r.ci95_low:+.5f}, {r.ci95_high:+.5f}]'] for r in sensitivity.itertuples()], 'lrr')
    paired = read('final_test_paired_comparisons')
    # Column names are defined by final_evaluation, and are checked here.
    rows = []
    for r in paired.to_dict('records'):
        rows.append([DISPLAY[r['comparator']], f'{r["log_loss_improvement"]:+.5f}', f'[{r["ci95_low"]:+.5f}, {r["ci95_high"]:+.5f}]'])
    em_paired = read('latent_em_paired_comparison').iloc[0]
    rows.append(['Latent BN', f'{-em_paired["log_loss_improvement"]:+.5f}', f'[{-em_paired["ci95_high"]:+.5f}, {-em_paired["ci95_low"]:+.5f}]'])
    table('test_intervals', ['Comparator', 'Hand advantage', '95\\% game interval'], rows, 'lrr')
    queries = read('inference_queries')
    qlabels = {'population': 'No evidence', 'close_two_tight_defender': r'$D<5,T=2,C<2$', 'close_two_open': r'$D<5,T=2,C\geq6$', 'long_three_moderate_defender': r'$25\leq D<30,T=3,4\leq C<6$'}
    table('queries', ['Evidence', 'Shots', 'Observed', 'BN query'], [[qlabels[r.query], f'{r.training_shots:,}', f'{r.empirical_make_rate:.3f}', f'{r.ve_make_probability:.3f}'] for r in queries.itertuples()], 'lrrr')
    sensitivity_summary = read('preprocessing_sensitivity_summary')
    rows = []
    schemes = [('original', 'Original six bins'),
               ('five_foot_25plus', 'Five-foot, 25+'),
               ('five_foot_30plus', 'Five-foot, 30+ (used)'),
               ('training_quantiles_7', 'Seven training quantiles')]
    distance = sensitivity_summary.query("diagnostic == 'distance_bins'").set_index(['configuration', 'model'])
    for configuration, label in schemes:
        rows.append([label, *[f'{distance.loc[(configuration, model), "mean_log_loss"]:.5f}' for model in ['Hand BN', 'Logistic regression', 'Naive Bayes']]])
    table('preprocessing', ['Distance scheme', 'Hand BN', 'Logistic', 'Naive Bayes'], rows, 'lrrr')
    # Use the committed copy, so report compilation needs no local NBA dataset.
    prep = json.loads((TABLES / 'final_test_metadata.json').read_text())['preprocessing']
    runtime = read('inference_runtime')
    runs = read('structure_stability_runs')
    chrono_meta = json.loads((TABLES / 'chronological_metadata.json').read_text())
    players = read('player_shot_quality').set_index('player_name')
    player_rows = []
    for name in ['Stephen Curry', 'Kyle Korver', 'Klay Thompson']:
        player = players.loc[name]
        player_rows.append([name, f'{int(player["shots"]):,}', f'{100*player["actual_fg"]:.1f}\\%', f'{100*player["expected_fg"]:.1f}\\%', f'{100*player["fg_over_expected"]:+.1f}'])
    table('players', ['Player', 'Shots', 'Actual FG', 'Expected FG', 'Gap (pp)'], player_rows, 'lrrrr')
    stability = read('player_split_half_stability').iloc[0]
    restarts = read('latent_em_restarts')
    macros = {
        'RawN': f'{prep["raw_rows"]:,}', 'CleanN': f'{prep["clean_rows"]:,}',
        'RemovedN': f'{prep["removed_rows_total"]:,}',
        'FitN': f'{prep["splits"]["train_fit"]["shots"]:,}',
        'ValN': f'{prep["splits"]["validation"]["shots"]:,}',
        'TestN': f'{prep["splits"]["test"]["shots"]:,}',
        'TrainN': f'{prep["splits"]["train"]["shots"]:,}',
        'ChronoTrainN': f'{chrono_meta["cohorts"]["training"]["shots"]:,}',
        'ChronoTestN': f'{chrono_meta["cohorts"]["evaluation"]["shots"]:,}',
        'BNFitSeconds': f'{runtime.iloc[0]["seconds"]:.3f}',
        'QueryMinMS': f'{runtime.iloc[1:]["seconds"].min()*1000:.1f}',
        'QueryMaxMS': f'{runtime.iloc[1:]["seconds"].max()*1000:.1f}',
        'SearchMinSeconds': f'{runs.learning_seconds.min():.2f}',
        'SearchMaxSeconds': f'{runs.learning_seconds.max():.2f}',
        'EMK': str(int(em['k'])),
        'EMValLL': f'{em_validation["val_log_loss"]:.5f}',
        'EMTestLL': f'{em["log_loss"]:.5f}',
        'EMHandAdvantage': f'{-em_paired["log_loss_improvement"]:.5f}',
        'EMHandLow': f'{-em_paired["ci95_high"]:.5f}',
        'EMHandHigh': f'{-em_paired["ci95_low"]:.5f}',
        'EMRestartN': str(len(restarts)),
        'EMConvergedN': str(int(restarts['converged'].sum())),
        'EMCappedN': str(int(restarts['hit_iteration_cap'].sum())),
        'EMFinalIter': str(int(em['iterations'])),
        'PlayerN': str(int(players['shots'].ge(200).sum())),
        'PlayerHalfN': str(int(stability['players'])),
        'PlayerHalfR': f'{stability["correlation"]:.3f}',
        'PlayerHalfDate': str(stability['split_date']),
    }
    (OUT / 'numbers.tex').write_text('\n'.join('\\newcommand{\\'+key+'}{'+value+'}' for key,value in macros.items())+'\n')
    for path in [Path(__file__).resolve(), OUT.parent / 'main.tex', OUT.parent / 'neurips_2026.sty']:
        inputs[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        'schema_version': 1,
        'description': 'Verified report inputs; detailed per-shot files are excluded.',
        'inputs_sha256': dict(sorted(inputs.items())),
        'generated_tables_sha256': {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(OUT.glob('*.tex'))
        },
    }
    (OUT / 'input_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('Generated report tables and numbers from verified result artifacts.')


if __name__ == '__main__':
    main()
