"""Naive Bayes combination, historical joint rates, and nesting checks."""

import numpy as np
import pandas as pd
import pytest
import torch

from alphaverify.domain import barrier, combination
from alphaverify.infrastructure.artifact_history import feature_bins_from_artifact


def logit(p):
    return np.log(p / (1 - p))


def test_no_condition_is_the_smoothed_baseline_and_one_condition_is_its_own_rate():
    hits, counts = np.array([[10., 40.]]), np.array([100., 100.])
    baseline = combination.smoothed_probability(hits, counts[None, :])
    np.testing.assert_allclose(combination.naive_bayes_probability(hits, counts, [], [])[0], baseline)
    condition_hits, condition_counts = np.array([[30., 5.]]), np.array([60., 60.])
    np.testing.assert_allclose(
        combination.naive_bayes_probability(hits, counts, [condition_hits], [condition_counts])[0],
        (condition_hits + 1) / (condition_counts[None, :] + 2),
    )


def test_two_conditions_add_their_log_odds_ratios():
    hits, counts = np.array([[20.]]), np.array([98.])
    first, second = (np.array([[40.]]), np.array([58.])), (np.array([[9.]]), np.array([48.]))
    base, p1, p2 = 21 / 100, 41 / 60, 10 / 50
    expected = 1 / (1 + np.exp(-(logit(base) + (logit(p1) - logit(base)) + (logit(p2) - logit(base)))))
    actual, support = combination.naive_bayes_probability(
        hits, counts, [first[0], second[0]], [first[1], second[1]]
    )
    np.testing.assert_allclose(actual, [[expected]])
    np.testing.assert_array_equal(support, [48.])


def test_thin_baseline_or_condition_cells_are_unsupported():
    hits, counts = np.zeros((1, 2)), np.array([100., 29.])
    result, support = combination.naive_bayes_probability(
        hits, counts, [np.zeros((1, 2))], [np.array([29., 100.])]
    )
    assert np.isnan(result).all()
    np.testing.assert_array_equal(support, [29., 29.])
    with pytest.raises(ValueError):
        combination.naive_bayes_probability(hits, counts, [np.zeros((2, 2))], [counts])


def history(n=1200, seed=4):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, .015, n)))
    return pd.DataFrame({"open": close, "high": close * 1.008, "low": close * .992,
                         "close": close, "volume": np.ones(n)})


def test_joint_rate_matches_stage1_for_all_bars_and_for_one_condition():
    data = history()
    barriers, horizons = np.array([-.02, -.01, .01, .02]), np.array([1, 5])
    outcomes = barrier.observed_outcomes(data, barriers, horizons)
    eligible = (np.isfinite(outcomes["downside_excursion"][horizons - 1])
                & np.isfinite(outcomes["upside_excursion"][horizons - 1]))
    rate, counts = combination.joint_touch_rate(
        np.ones(len(data), dtype=bool), outcomes["touch_mask"], eligible
    )
    np.testing.assert_array_equal(rate, outcomes["baseline_probability"])

    feature = pd.Series(np.sin(np.arange(len(data)) / 7))
    edges = barrier.bin_edges(feature, 4)
    cube = barrier.touch_tensor(data, feature, horizons, barriers, edges)
    bins = feature_bins_from_artifact({"feature_values": feature.to_numpy(), "bin_edges": edges})
    rate, counts = combination.joint_touch_rate(bins == 2, outcomes["touch_mask"], eligible)
    np.testing.assert_allclose(rate, cube["conditional_probability"][:, 2, :], rtol=0, atol=1e-12)
    np.testing.assert_array_equal(counts, cube["bin_observation_counts"][2])


def test_nesting_violations_follow_barrier_distance_and_horizon():
    barriers, horizons = np.array([-.02, -.01, .01, .02]), np.array([1, 5])
    coherent = np.array([[.1, .2], [.3, .4], [.3, .4], [.1, .2]])
    assert combination.nesting_violations(coherent, barriers, horizons) == 0
    broken = coherent.copy()
    broken[0] = [.35, .45]        # farther downside barrier above nearer one at both horizons: 2
    broken[3, 1] = .05            # longer horizon below shorter one: 1
    broken[2, 0] = np.nan         # unsupported cells are ignored
    assert combination.nesting_violations(broken, barriers, horizons) == 3
    order = [2, 0, 3, 1]
    assert combination.nesting_violations(broken[order], barriers[order], horizons) == 3


def test_feature_bins_use_stored_edges_the_stage1_tie_convention_and_missing_values():
    edges = np.array([.5, 1.5])
    values = np.array([.2, .5, 1.0, 1.5, 9.0, np.nan])
    bins = feature_bins_from_artifact({"feature_values": values, "bin_edges": edges})
    assert bins.tolist() == [0, 0, 1, 1, 2, -1]
    expected = barrier.bin_indices(torch.from_numpy(values[None, :5]), torch.from_numpy(edges[None]))
    assert bins[:5].tolist() == expected[0].tolist()
