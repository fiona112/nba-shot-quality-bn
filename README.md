
# NBA Shot Quality Prediction with Bayesian Networks

**CSE 150A/250A Final Project — Group HALF**

Team: Anna He, Fiona Feng, Hyeonung Cho, Logan Szeto

## Overview

We estimate the probability that an NBA shot is made based
on shot circumstances, without using shooter or defender
identity as predictive features.

Our goal is to model shot quality separately from observed
shooting performance.

## Dataset

NBA Shot Logs (2014–15 season)

https://www.kaggle.com/datasets/dansbecker/nba-shot-logs

Predictor variables:
- Shot type (2PT / 3PT)
- Shot distance
- Closest defender distance
- Shot clock
- Dribbles
- Touch time

Target: made or missed shot.

The dataset is not committed to this repository.
Download `shot_logs.csv` and put it in `data/raw/`.

### Dataset Setup

The NBA Shot Logs dataset is kept out of Git, along with all processed CSV files.
Download the [Kaggle NBA Shot Logs dataset](https://www.kaggle.com/datasets/dansbecker/nba-shot-logs),
extract `shot_logs.csv`, and place it in `data/raw/`.

After installing the dependencies, run preprocessing from the repository root:

```bash
python -m src.preprocess
```

Preprocessing audits and cleans the raw data, bins the six predictor variables,
and creates all four experiment inputs in `data/processed/`:

| File | Purpose |
| --- | --- |
| `train.csv` | Initial training set: 80% of games, split seed 42 |
| `test.csv` | Held-out test set: 20% of games, split seed 42 |
| `train_fit.csv` | Model-selection training set: 80% of the initial training games, split seed 123 |
| `validation.csv` | Validation set: 20% of the initial training games, split seed 123 |

Games do not overlap between `train_fit.csv`, `validation.csv`, and `test.csv`.
The initial `train.csv` contains both the training-fit and validation subsets.
Percentages refer to games; shot counts vary by game. Rerunning preprocessing
reproduces the same splits and replaces the processed files. A cleaning and split
summary is saved as `data/processed/preprocessing_report.json`.

Every retained shot and clock-exclusion row includes `RAW_ROW_ID`, its zero-based
position among the parsed rows in the raw CSV (excluding the header). It is
assigned before cleaning and preserved in all four splits. Use the pair
`(raw-file SHA256, RAW_ROW_ID)` to identify a shot across experiment outputs;
reordering or replacing the raw file changes that identity. `RAW_ROW_ID` and
`GAME_ID` are metadata, not model predictors.

The preprocessing report also records:

- The raw file's SHA256 checksum and size, plus checksums of the four split CSVs
  and the clock-exclusion CSV.
- Git commit and dirty-worktree status, Python/platform details, and exact
  installed package versions. Git values are `null` for exported source trees
  or when Git is unavailable.
- SHA256 checksums of Python source files and `requirements.txt`, so an
  uncommitted code change is distinguishable from its base commit.
- Predictor names, all bin edges/labels/source columns, clock policies, and the
  game-level split seeds/fractions and resulting shot/game counts.

The report uses strict JSON: infinite bin boundaries are written as the strings
`"-inf"` and `"inf"`. Its UTC timestamp changes on each run; split CSV checksums
should match when data, code, settings, and package versions match. Requirements
specify compatible version ranges; the report captures the actual environment.
Keep the report with any experiment outputs you share so their inputs can be
checked. Raw data, processed data, and `.venv` remain excluded from Git.

To check preprocessing and provenance without downloading the dataset:

```bash
python -m pytest tests/test_preprocess.py tests/test_provenance.py -v
```

Shot-clock preprocessing uses the following policy:

- For a missing `SHOT_CLOCK` with a valid game clock between 0 and 24 seconds
  inclusive, retain the shot and use the game clock as `SHOT_CLOCK_EFFECTIVE`.
  Preserve the missing `SHOT_CLOCK`, set `SHOT_CLOCK_WAS_MISSING`, and label
  `SHOT_CLOCK_SOURCE` as `period_clock_proxy`. The game clock is a proxy for
  time available, not a recovered shot-clock reading: a missing value near the
  end of a period does not prove that the shot clock was off.
- Remove shots with missing clocks outside that window, including missing clocks
  whose game clock cannot be parsed. Also remove invalid recorded shot clocks;
  do not reinterpret invalid recorded values as clock-off cases.
  Exclusion happens before all dataset splits. Save these rows separately in
  `data/processed/excluded_shot_clock.csv`, with `EXCLUDE_SHOT_CLOCK = True`,
  a reason, and a zero-based `RAW_ROW_ID` identifying the original CSV row.
  This audit file is not a model input; the original raw CSV is preserved.
- Keep recorded `24.0` values and flag them using `SHOT_CLOCK_IS_24` and
  `SHOT_CLOCK_SOURCE = recorded_24_unverified`. Putback context alone cannot
  verify that the clock was exactly 24 at release. No automatic putback label
  is inferred from touch time, dribbles, or distance.
- Preserve valid observed clocks even near the end of a period. Model feature
  `S` is binned from `SHOT_CLOCK_EFFECTIVE`. Clock-source flags are audit metadata
  and are not included among the model predictors.

Of the 5,567 raw missing shot clocks, 3,554 fall in the 0-to-24-second window
(including four at exactly 24 seconds), and 2,013 fall outside it. Other cleaning
rules may remove additional rows in either group; the report gives final counts.

Shot distance (`D`) uses seven buckets in feet: `<5`, `5-10`, `10-15`,
`15-20`, `20-25`, `25-30`, and `30+`. Each interval includes its lower
boundary and excludes its upper boundary; exactly 30 feet belongs to `30+`.
Shot type (`T`) remains the recorded two- or three-point value, independently
of distance. Closest-defender distance (`C`) keeps the NBA categories:
`0-2` (Very Tight), `2-4` (Tight), `4-6` (Open), and `6+` (Wide Open).

After changing buckets, regenerate processed data and rerun the experiments
before using the saved result tables; existing tables may use earlier buckets.

### Running the Experiments

Run the validation experiments in this order after preprocessing:

```bash
PYTHONHASHSEED=0 python -m src.structure_learning
PYTHONHASHSEED=0 python -m src.ablation
PYTHONHASHSEED=0 python -m src.calibration
```

These three commands use only `train_fit.csv` and `validation.csv`. They verify
the input checksums against `preprocessing_report.json` and reject overlapping
shot/game identifiers. Ablation and calibration load the current learned graph
and require its provenance to match those inputs. `PYTHONHASHSEED=0` fixes Python
set iteration for hill-climbing ties; the value is recorded with package versions
and source hashes. Equivalent-score graph orientations may otherwise differ.
The dependency range keeps the existing pgmpy estimator interface; this run was
verified with pgmpy 1.1.2.

Each command saves a JSON sidecar in `results/tables/`:
`structure_validation_metadata.json`, `ablation_validation_metadata.json`, or
`calibration_metadata.json`. These include input counts/checksums, a copy of the
preprocessing report, model settings, code/environment provenance, dependencies,
and output checksums. The metadata travels with the results even though the
processed data is excluded from Git. `validation_predictions.csv` includes
`RAW_ROW_ID`, `GAME_ID`, the target (`actual`), and all five models' probabilities.

Calibration regenerates both `calibration_comparison.png` and
`calibration_bin_counts.png` from the same validation shots. The metrics tables
include training/validation shot counts. The shared hand-BN metrics should agree
across structure, ablation, and calibration; the learned-BN metrics should also
agree across all three. Results preceding this synchronization remain available
in Git at commit `d17a9340257877eb52208853b9b2de680d23fc42`.

Additional commands for the existing test-set comparisons, inference checks,
and chronological robustness experiment are:

```bash
python -m src.baselines
python -m src.naive_bayes
python -m src.bayesian_network
python -m src.variable_elimination
python -m pytest tests/test_variable_elimination.py -v
PYTHONHASHSEED=0 python -m src.chronological_evaluation
```

Use validation for model selection. The chronological experiment uses a separate
date-based partition, so its metrics should be labeled separately from validation
and the random-game test results. Run it separately when regenerating those
outputs; rerunning the three validation commands does not update chronology or
test-set result files.

Use Python 3.11 or 3.12 and install dependencies with
`python -m pip install -r requirements.txt`.

### Bucket and Shot-Clock Diagnostics

```bash
python -m src.compare_distance_bins
python -m src.audit_shot_clock
```

Both commands use five game-level validation splits of `train.csv`, including
the existing validation seed 123. They do not read `test.csv`. Validation sets
overlap, so their averages are sensitivity checks rather than independent trials.
Distance comparisons include the original buckets, five-foot buckets with and
without a separate `30+` state, and seven quantile buckets fitted only on each
split's training shots. The chosen cutoffs are recorded with each result.

The raw dataset has 3,857 shots with `SHOT_CLOCK == 24.0` (3.01%). Many have
putback-like characteristics, but individually checked plays also include longer
possessions. In 2014-15, offensive rebounds could reset the clock to 24 seconds;
the NBA changed that reset to 14 seconds in 2018. Archived NBA gamebooks confirm
several rebound sequences and several non-putback examples, but do not report
the exact shot clock at release. Dataset game-clock timestamps also differ from
play-by-play timestamps, so game, player, attempt order, period, and outcome were
used to cross-check selected shots.

The clock diagnostic uses the processed data after the missing-clock policy
above, and compares the primary policy for recorded 24.0 values with a separate
categorical state for exactly 24 and with 24 treated as missing. These sensitivity
runs only modify the categorical `S` input in memory; they do not rerun the row
removal policy or modify the saved raw/effective clocks. The simple hand-designed
make CPT does not use shot clock, so its predictions are identical under the
three clock policies.

Results are saved in `results/tables/distance_binning_validation.csv`,
`shot_clock_audit_summary.csv`, and `shot_clock_policy_validation.csv`.
`shot_clock_verified_examples.csv` records seven manually checked examples with
official gamebook URLs and page numbers. These selected cases are diagnostic
examples, not an estimate of how many 24.0 values are valid or erroneous.

For a final comparison on identical cleaned training and test shots, run
`python -m src.compare_preprocessing`. This compares the original distance
buckets with a missing-clock category, the new distance buckets with that same
clock handling, and the new buckets with period-clock proxies. All three respect
the unexplained missing-clock exclusions. Results go to
`results/tables/preprocessing_comparison_test.csv`. This command uses the test
set for final reporting; use validation comparisons for further model selection.

## Models

1. Constant league-average probability
2. Logistic regression
3. Naive Bayes
4. Hand-designed Bayesian network
5. Structure-learned Bayesian network
6. Optional latent shot-difficulty Bayesian network

## Evaluation

- Held-out log loss
- Brier score
- Accuracy
- Calibration curves
- Conditional-independence analyses
- Model structure and feature ablations

Games are separated between training and testing.

## Setup

Use Python 3.11 or 3.12. For example, with Python 3.12 installed:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run

```bash
python -m src.preprocess
python -m src.baselines
```

## Limitations

Because shooter identity is excluded, the model estimates
make probability among observed shot attempts. These shots
are selected by actual NBA players, not randomly assigned.

Better shooters may attempt more difficult shots, creating
selection effects that limit the interpretation of predicted
shot quality as a universal league-average shooting ability.

The dataset also lacks precise shot coordinates and detailed
shot-type labels.

## AI Assistance

We will document AI-assisted code and writing contributions
according to course requirements.

### Graph assumptions

`python -m src.graph_assumptions` verifies selected d-separation claims for the
hand BN, Naive Bayes, and current learned graph. The hand graph is defined once
in `src/graphs.py`, shared by CPT prediction, full inference, and comparison.
`graph_assumptions.csv` lists the restrictions each model imposes, including the
joint outcome claim `M independent of {S,B} given {D,T,C,H}`. These are model
assumptions, not evidence that observed shots satisfy independence or causality.
The learned outcome claims use the verified graph's actual parent set.

### Empirical conditional-dependence diagnostics

`python -m src.conditional_independence` computes plug-in conditional mutual
information (CMI, in nats) on training-fit shots for the declared graph claims.
It reports complete-case counts, effective/empty table cells, expected-count
support, and the fraction of shots in sparse or constant strata. The descriptive
`G2 = 2*N*CMI` statistic is not assigned a chi-square p-value: shots are clustered
within games/players, and high-dimensional tables can be sparse. Positive CMI is
upward biased in finite samples and does not by itself reject a graph assumption.
Joint claims are measured jointly, rather than inferred from pairwise checks.

### Predictive sensitivity and clustered uncertainty

`python -m src.ci_sensitivity` compares matched make-parent sets on the same
validation shots. Positive log-loss improvement favors adding the feature.
Intervals use 1,000 seeded paired game-cluster bootstrap draws: sample games
with replacement and retain every shot from each sampled game, weighting shots
equally within each replicate. These are exploratory intervals conditional on
fitted models and prior validation selection; they neither prove conditional
independence nor include refitting/selection uncertainty or cross-game player
dependence. Compact tables/figures are committed; detailed identified predictions
stay in ignored `data/processed/ci_validation_predictions.csv` and can be rebuilt.

### Frozen final comparison

`PYTHONHASHSEED=0 python -m src.final_evaluation` verifies the outer training/test
files, refits the five prespecified models on all outer-training shots, and
reports a common random-game test comparison. It separately relearns/refits on
chronologically earlier dates and evaluates later dates. The original hand
parents and alpha are retained; no test outcome selects a model, graph,
threshold, or tuning parameter. Paired primary-hand versus comparator intervals
resample complete games using the same fixed-fit bootstrap as the sensitivity
analysis. The test set was inspected by earlier scripts, and chronology reuses
this same dataset; neither is described as a pristine external holdout.

`final_test_*` and `chronological_*` summaries, calibration figures and provenance
are saved in `results/`; large per-shot prediction CSVs are written only to ignored
`data/processed/`. The final evaluator is the documented way to regenerate both
cohorts together; the original chronological module remains a compatibility entry
point without the full comparison/provenance checks.

### Graph interpretation and exact inference

`PYTHONHASHSEED=0 python -m src.interpretation` produces the actual hand/learned
DAGs, observed make-CPT rows with counts and marginal Beta-prior intervals,
supported partial-evidence queries, single-run wall-clock timings, and three
prespecified subsamples of complete training-fit games. These runs never use
final-test shots. Posterior CPT intervals use a row-wise independent-shot
likelihood; they are not game-cluster intervals. Subsample edge frequencies are
exploratory stability descriptions, not confidence levels or causal evidence.

Core variable-elimination tests now use synthetic data, so they run without the
NBA CSV. The full test suite is `python -m pytest -q` after dependencies are
installed. Inference complexity grows with intermediate-factor size and CPT
parent cardinalities; greedy structure learning returns a local solution rather
than a guarantee of the globally optimal graph.
