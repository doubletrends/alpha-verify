"""False discovery rate adjustment across condition-bin tests."""

import numpy as np
import pytest

from alphaverify.domain.multiple_testing import benjamini_hochberg


def reference_q_values(p_values):
    """Step-up definition: q_i = min over p_(j) >= p_i of m * p_(j) / j, capped at 1."""
    ordered = sorted(p_values)
    m = len(ordered)
    return [
        min(1.0, min(m * ordered[j] / (j + 1) for j in range(m) if ordered[j] >= p))
        for p in p_values
    ]


def test_hand_worked_example_in_input_order():
    # Sorted p: .01 .02 .03 .20 -> m*p/j = .04 .04 .04 .20
    np.testing.assert_allclose(benjamini_hochberg([.20, .01, .03, .02]), [.20, .04, .04, .04])


def test_matches_step_up_definition_with_ties_and_the_monte_carlo_floor():
    rng = np.random.default_rng(5)
    p = np.round(rng.uniform(0, 1, 300) ** 3, 3)
    p[:12] = 1 / 1001
    rng.shuffle(p)
    q = benjamini_hochberg(p)
    np.testing.assert_allclose(q, reference_q_values(p.tolist()), rtol=0, atol=1e-12)
    for value in np.unique(p):
        assert np.unique(q[p == value]).size == 1
    order = np.argsort(p, kind="stable")
    assert np.all(np.diff(q[order]) >= 0)
    assert np.all(q >= p) and np.all(q <= 1)


def test_empty_and_invalid_inputs():
    assert benjamini_hochberg([]).shape == (0,)
    for bad in ([.1, np.nan], [-.1], [1.2], [[.1]]):
        with pytest.raises(ValueError):
            benjamini_hochberg(bad)
