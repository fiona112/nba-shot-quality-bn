from itertools import product

import numpy as np
import pandas as pd
import pytest

from src.conditional_independence import conditional_mutual_information as cmi


def test_independence_and_perfect_dependence_have_known_information():
    independent = pd.DataFrame(list(product([0, 1], [0, 1])) * 10, columns=['X', 'Y'])
    assert cmi(independent, ['X'], ['Y'])['cmi_nats'] == pytest.approx(0, abs=1e-12)
    dependent = pd.DataFrame({'X': [0, 1] * 20, 'Y': [0, 1] * 20})
    assert cmi(dependent, ['X'], ['Y'])['cmi_nats'] == pytest.approx(np.log(2))


def test_conditioning_removes_common_cause_association():
    data = pd.DataFrame({'X': [0, 1] * 20, 'Y': [0, 1] * 20, 'Z': [0, 1] * 20})
    assert cmi(data, ['X'], ['Y'])['cmi_nats'] > 0.6
    result = cmi(data, ['X'], ['Y'], ['Z'])
    assert result['cmi_nats'] == pytest.approx(0)
    assert result['shots_in_constant_strata_fraction'] == 1


def test_joint_claim_detects_xor_that_pairwise_claims_miss():
    data = pd.DataFrame(list(product([0, 1], [0, 1])) * 25, columns=['B', 'S'])
    data['M'] = data.B ^ data.S
    assert cmi(data, ['M'], ['B'])['cmi_nats'] == pytest.approx(0)
    assert cmi(data, ['M'], ['S'])['cmi_nats'] == pytest.approx(0)
    assert cmi(data, ['M'], ['B', 'S'])['cmi_nats'] == pytest.approx(np.log(2))
    assert cmi(data.sample(frac=1, random_state=3), ['M'], ['S', 'B'])['cmi_nats'] == pytest.approx(np.log(2))


def test_missing_and_sparse_support_are_reported():
    data = pd.DataFrame({'X': [0, 1, None], 'Y': [0, 1, 0]})
    result = cmi(data, ['X'], ['Y'])
    assert result['shots'] == 2 and result['dropped_missing_shots'] == 1
    assert result['empty_joint_cells'] == 2
    assert result['expected_cells_below_5'] == 4
    assert result['shots_in_sparse_informative_strata_fraction'] == 1
    with pytest.raises(ValueError, match='No complete'):
        cmi(data.iloc[2:], ['X'], ['Y'])
    with pytest.raises(ValueError, match='disjoint'):
        cmi(data, ['X'], ['X'])
