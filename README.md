
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

```bash
python -m src.baselines
python -m src.naive_bayes
python -m src.bayesian_network
python -m src.variable_elimination
python -m pytest tests/test_variable_elimination.py -v
python -m src.structure_learning
python -m src.ablation
```

Use Python 3.11 and install dependencies with `pip install -r requirements.txt`.

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

Python 3.11 or 3.12 recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
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
