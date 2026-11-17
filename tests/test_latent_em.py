
import numpy as np
import pandas as pd

from src.latent_em import LatentShotDifficultyBN
from src.structure_learning import FEATURES


TRUE_P_MAKE = [0.7, 0.3]


def synthetic_shots(n=4000, seed=0):
    """
    Sample from a two-state latent-class model where each
    state strongly prefers one value of every feature.
    """
    rng = np.random.default_rng(seed)
    q = rng.integers(0, 2, size=n)

    data = {}
    for var in FEATURES:
        # 90% of the time the feature reveals the hidden state.
        reveal = rng.random(n) < 0.9
        data[var] = np.where(reveal, q, 1 - q).astype(str)

    data["M"] = (
        rng.random(n) < np.take(TRUE_P_MAKE, q)
    ).astype(int).astype(str)

    return pd.DataFrame(data)


def test_em_objective_never_decreases():
    model = LatentShotDifficultyBN(k=3, seed=1).fit(synthetic_shots())
    objective = np.array([h["objective"] for h in model.history])

    assert len(objective) > 2
    assert np.all(np.diff(objective) >= -1e-6)


def test_em_recovers_hidden_make_rates():
    model = LatentShotDifficultyBN(k=2, seed=0).fit(synthetic_shots())
    learned = sorted(model.describe_states()["p_make"])

    assert np.allclose(learned, sorted(TRUE_P_MAKE), atol=0.05)


def test_predictions_are_valid_probabilities():
    data = synthetic_shots()
    model = LatentShotDifficultyBN(k=2, seed=0).fit(data)

    # A value never seen in training is summed out, not an error.
    unseen = data.head(5).copy()
    unseen["D"] = "never-seen"

    for frame in (data, unseen):
        p = model.predict_proba(frame)
        assert np.all((p > 0) & (p < 1))
