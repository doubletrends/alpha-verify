"""Numerical parity across measurement, selection, and null validation."""

import numpy as np
import pandas as pd
import pytest
import torch

from barrierlab.domain import barrier, scoring, selection, shift, validation


def history(n=420, seed=7):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.002, 0.02, n)))
    open_ = np.r_[close[0], close[:-1]]
    return pd.DataFrame({
        "open": open_, "high": np.maximum(open_, close) * 1.01,
        "low": np.minimum(open_, close) * 0.99, "close": close,
        "volume": np.ones(n),
    })


def observed_cube(data, feature, n_bins, deltas, horizons):
    cube = barrier.touch_tensor(data, feature, horizons, deltas, barrier.bin_edges(feature, n_bins))
    baseline = barrier.touch_tensor(
        data, pd.Series(np.ones(len(data))), horizons, deltas, np.array([]),
    )["prob"][:, 0, :]
    return shift.from_cube(cube, baseline)


@pytest.mark.parametrize("kind,n_bins", [("continuous", 2), ("continuous", 10),
                                         ("ties", 10), ("constant", 10),
                                         ("missing", 4), ("sparse", 10)])
def test_same_history_has_same_selection_observed_and_null_score(kind, n_bins):
    data = history()
    x = np.linspace(-1, 1, len(data))
    if kind == "ties":
        x = np.arange(len(data)) % 3
    elif kind == "constant":
        x = np.ones(len(data))
    elif kind == "missing":
        x[:75] = np.nan
        x[100] = np.inf
    elif kind == "sparse":
        x[:250] = np.nan
    feature = pd.Series(x)
    deltas, horizons = np.array([-.06, -.02, .02, .06]), np.array([1, 7, 45, 420])
    cube = observed_cube(data, feature, n_bins, deltas, horizons)
    observed = validation.bin_score_from_shift_cube(cube)
    null = validation.batched_bin_score_sums(
        data.to_numpy()[None, :, :], x[None, :], deltas, horizons, n_bins,
    )[0]
    np.testing.assert_allclose(null[:len(observed)], observed, atol=1e-10)
    np.testing.assert_array_equal(null[len(observed):], 0)
    ranked = selection.rank_nodes(
        [{"id": "test", "family": "test", "feature": "close", "params": {}}],
        lambda _: cube, 100,
    )
    for row in ranked["candidates"]:
        assert row["score_pp"] == pytest.approx(observed[row["bin"]])


def test_market_drift_alone_scores_zero_in_both_paths():
    close = 100 * 1.02 ** np.arange(101)
    data = pd.DataFrame({"open": close, "high": close, "low": close,
                         "close": close, "volume": np.ones(101)})
    x = pd.Series(np.arange(101, dtype=float))
    cube = observed_cube(data, x, 2, np.array([-.01, .01]), np.array([1]))
    np.testing.assert_array_equal(validation.bin_score_from_shift_cube(cube), [0, 0])
    np.testing.assert_array_equal(validation.batched_bin_scores(
        data.to_numpy()[None], x.to_numpy()[None], .01, 1, 2,
    ), [[0, 0]])
    np.testing.assert_array_equal(validation.bin_scores(data, x, .01, 1, 2), [0, 0])


def test_bins_use_median_finite_values_and_observed_tie_convention():
    x = torch.tensor([[0., 1., 2., 3., 4., float("nan")]], dtype=torch.float64)
    edges = barrier.batched_bin_edges(x, 2)
    assert edges.tolist() == [[2.0]]
    assert barrier.bin_indices(x[:, :5], edges).tolist() == [[0, 0, 0, 1, 1]]


def test_batch_uses_each_paths_own_baseline_and_edges():
    frames = [history(seed=3), history(seed=31)]
    features = [pd.Series(np.linspace(-3, 1, 420)), pd.Series(np.sin(np.arange(420)))]
    deltas, horizons = np.array([-.03, .03]), np.array([1, 5])
    expected = [validation.bin_score_from_shift_cube(observed_cube(d, x, 3, deltas, horizons))
                for d, x in zip(frames, features)]
    actual = validation.batched_bin_score_sums(
        np.stack([d.to_numpy() for d in frames]), np.stack(features), deltas, horizons, 3,
    )
    np.testing.assert_allclose(actual, expected, atol=1e-10)
    assert not np.allclose(actual[0], actual[1])


def test_fixed_external_edges_preserve_collapsed_observed_bins():
    data = history()
    x = pd.Series(np.arange(len(data), dtype=float) % 3)
    deltas, horizons = np.array([-.03, .03]), np.array([7])
    cube = observed_cube(data, x, 10, deltas, horizons)
    actual = validation.batched_bin_score_sums(
        data.to_numpy()[None], x.to_numpy()[None], deltas, horizons, 10, edges=cube["edges"],
    )
    np.testing.assert_allclose(actual[0], validation.bin_score_from_shift_cube(cube), atol=1e-10)


def test_shared_scorer_excludes_thin_bins_even_with_finite_probabilities():
    prob = np.array([[[.2], [.3], [.0]], [[.7], [.6], [1.]]])
    base = np.array([[.3], [.6]])
    scores = scoring.score_grid(prob, base, np.array([[30], [30], [29]]), [-.1, .1])
    np.testing.assert_allclose(scores["total"], [20, 0, 0], atol=1e-12)
    scores = scoring.score_grid(prob, base, np.array([[30], [29], [29]]), [-.1, .1])
    np.testing.assert_array_equal(scores["total"], [0, 0, 0])


def test_unpaired_grid_has_zero_score():
    result = scoring.score_grid(np.ones((1, 2, 1)), np.ones((1, 1)),
                                np.full((2, 1), 100), [.1])
    np.testing.assert_array_equal(result["total"], [0, 0])
