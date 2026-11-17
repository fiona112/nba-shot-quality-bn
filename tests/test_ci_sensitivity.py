import numpy as np
import pytest

from src.ci_sensitivity import log_losses, paired_game_bootstrap


def test_log_loss_sign_and_probability_validation():
    reduced = log_losses([0, 1], [0.5, 0.5])
    expanded = log_losses([0, 1], [0.1, 0.9])
    assert (reduced - expanded > 0).all()
    np.testing.assert_allclose(expanded, -np.log(0.9))
    with pytest.raises(ValueError, match='matching shapes'):
        log_losses([0, 1], [0.5])
    with pytest.raises(ValueError, match='lie in'):
        log_losses([0], [-0.1])
    with pytest.raises(ValueError, match='matching shapes'):
        log_losses([0.5], [0.5])


def test_cluster_resampling_preserves_within_game_pairs_and_shot_weights():
    # One game has three +1 losses; the other has one +3 loss.
    losses = np.array([1., 1., 1., 3.])
    games = np.array([10, 10, 10, 20])
    result = paired_game_bootstrap(losses, games, replicates=1000, seed=7)
    assert result['log_loss_improvement'] == 1.5
    assert result['ci95_low'] == 1 and result['ci95_high'] == 3
    assert result['games'] == 2 and result['shots'] == 4
    order = [3, 1, 0, 2]
    assert paired_game_bootstrap(losses[order], games[order], 1000, 7) == result
    assert paired_game_bootstrap(losses, games, 1000, 7) == result


def test_degenerate_perfect_pairing_and_invalid_inputs():
    result = paired_game_bootstrap(np.zeros(6), [1, 1, 2, 2, 3, 3], 25, 3)
    assert result['ci95_low'] == result['ci95_high'] == 0
    with pytest.raises(ValueError, match='two games'):
        paired_game_bootstrap([1, 2], [1, 1])
    with pytest.raises(ValueError, match='aligned'):
        paired_game_bootstrap([1], [1, 2])
    with pytest.raises(ValueError, match='Finite'):
        paired_game_bootstrap([1, np.nan], [1, 2])
