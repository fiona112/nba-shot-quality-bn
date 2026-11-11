
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
