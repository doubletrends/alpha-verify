"""Numerical parity across measurement, bin scoring, and null validation."""

import numpy as np
import pandas as pd
import pytest
import torch

from barrierlab.domain import barrier, scoring, shift, validation


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
    )["conditional_probability"][:, 0, :]
    return shift.from_cube(cube, baseline)


def observed_scores(cube):
    return scoring.bin_scores(
        cube["conditional_probability"], cube["baseline_probability"],
        cube["bin_observation_counts"], cube["barriers"],
    ).bin_score.cpu().numpy()


@pytest.mark.parametrize("kind,n_bins", [("continuous", 2), ("continuous", 10),
                                         ("ties", 10), ("constant", 10),
                                         ("missing", 4), ("sparse", 10)])
def test_stage1_full_grid_matches_streamed_scores_and_validity(kind, n_bins):
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
    observed = observed_scores(cube)
    result = validation.score_histories(
        data.to_numpy()[None, :, :], x[None, :], deltas, horizons, n_bins,
    )
    null = result.bin_score[0]
    np.testing.assert_allclose(null[:len(observed)], observed, atol=1e-10)
    np.testing.assert_array_equal(null[len(observed):], 0)
    expected_valid = scoring.bin_scores(
        cube["conditional_probability"], cube["baseline_probability"],
        cube["bin_observation_counts"], deltas,
    ).score_supported
    np.testing.assert_array_equal(result.score_supported[0, :len(observed)], expected_valid)
    assert not result.score_supported[0, len(observed):].any()


def test_cached_and_streamed_measurement_agree_with_missing_prices():
    data = history(n=100)
    data.loc[15, "high"] = np.nan
    feature = pd.Series(np.arange(100, dtype=float))
    feature[:10] = np.nan
    # Unsorted/repeated horizons exercise cache lookup and ladder restarts.
    horizons, deltas = np.array([7, 1, 7, 100, 3]), np.array([-.02, .02])
    edges = barrier.bin_edges(feature, 2)
    direct = barrier.touch_tensor(data, feature, horizons, deltas, edges)
    cached = barrier.touch_tensor(data, feature, horizons, deltas, edges,
                                  excursions=barrier.forward_extremes_upto(data, 100))
    for key in ("conditional_probability", "bin_hit_counts",
                "bin_observation_counts", "eligible_observation_count"):
        np.testing.assert_array_equal(direct[key], cached[key])


def test_measurement_baseline_includes_feature_warmup_and_uses_float64():
    data = pd.DataFrame({"close": [100.] * 5, "high": [100., 101., 110., 103., 120.],
                         "low": [100., 99., 90., 98., 95.]})
    features = np.array([[np.nan, np.nan, 0., 1., 1.]])
    edges, slices = barrier.measure_histories(barrier.ohlcv_tensor(data).float(), features,
                                              [-.05, .05], [1], 2,
                                              bin_edges=[.5], min_n=1)
    measured = next(slices)
    assert measured.conditional_probability.dtype == torch.float64
    np.testing.assert_allclose(measured.baseline_probability, [[[.5], [.5]]])
    np.testing.assert_array_equal(measured.bin_observation_counts, [[[1], [1]]])
    np.testing.assert_array_equal(
        measured.conditional_probability, [[[[0.], [1.]], [[0.], [1.]]]]
    )


def test_market_drift_alone_scores_zero_in_both_paths():
    close = 100 * 1.02 ** np.arange(101)
    data = pd.DataFrame({"open": close, "high": close, "low": close,
                         "close": close, "volume": np.ones(101)})
    x = pd.Series(np.arange(101, dtype=float))
    cube = observed_cube(data, x, 2, np.array([-.01, .01]), np.array([1]))
    np.testing.assert_array_equal(observed_scores(cube), [0, 0])
    result = validation.score_histories(
        data.to_numpy()[None], x.to_numpy()[None], np.array([-.01, .01]), np.array([1]), 2,
    )
    np.testing.assert_array_equal(result.bin_score, [[0, 0]])
    np.testing.assert_array_equal(result.score_supported, [[True, True]])


def test_bins_use_median_finite_values_and_observed_tie_convention():
    x = torch.tensor([[0., 1., 2., 3., 4., float("nan")]], dtype=torch.float64)
    edges = barrier.batched_bin_edges(x, 2)
    assert edges.tolist() == [[2.0]]
    assert barrier.bin_indices(x[:, :5], edges).tolist() == [[0, 0, 0, 1, 1]]


def test_batch_uses_each_paths_own_baseline_and_edges():
    frames = [history(seed=3), history(seed=31)]
    features = [pd.Series(np.linspace(-3, 1, 420)), pd.Series(np.sin(np.arange(420)))]
    deltas, horizons = np.array([-.03, .03]), np.array([1, 5])
    expected = [observed_scores(observed_cube(d, x, 3, deltas, horizons))
                for d, x in zip(frames, features)]
    actual = validation.score_histories(
        np.stack([d.to_numpy() for d in frames]), np.stack(features), deltas, horizons, 3,
    )
    np.testing.assert_allclose(actual.bin_score, expected, atol=1e-10)
    assert not np.allclose(actual.bin_score[0], actual.bin_score[1])


def test_delta_matrix_matches_scalar_touch_rates():
    frames = [history(n=100, seed=seed) for seed in (2, 9)]
    paths = torch.as_tensor(np.stack([frame.to_numpy() for frame in frames]))
    feature = torch.stack((torch.arange(100), torch.arange(100).flip(0))).to(torch.float64)
    edges = barrier.batched_bin_edges(feature, 4)
    indices = barrier.bin_indices(feature, edges)
    deltas = np.array([-.07, -.02, 0., .02, .07])
    lo, hi = next(barrier.iter_extremes(paths, [7]))

    prob, hits, counts, observed = barrier.touch_rate_matrix(
        lo, hi, feature, indices, 4, deltas,
    )
    scalar = [barrier.touch_rates(lo, hi, feature, indices, 4, delta) for delta in deltas]
    torch.testing.assert_close(prob, torch.stack([row[0] for row in scalar], dim=1), equal_nan=True)
    torch.testing.assert_close(hits, torch.stack([row[1] for row in scalar], dim=1))
    torch.testing.assert_close(counts, scalar[0][2])
    torch.testing.assert_close(observed, scalar[0][3])


def test_persistable_observed_cache_matches_uncached_measurement():
    data = history(n=100)
    feature = pd.Series(np.sin(np.arange(100)))
    deltas, horizons = np.array([-.03, 0., .03]), np.array([1, 3, 7])
    edges = barrier.bin_edges(feature, 3)
    expected = barrier.touch_tensor(data, feature, horizons, deltas, edges)
    low, high = barrier.forward_extremes_upto(data, int(horizons.max()))
    lo, hi = low[horizons - 1], high[horizons - 1]
    delta_matrix = deltas.reshape(1, 1, -1)
    ok = np.isfinite(lo) & np.isfinite(hi)
    touches = np.where(
        delta_matrix < 0, lo[:, :, None] <= delta_matrix, hi[:, :, None] >= delta_matrix,
    ) & ok[:, :, None]
    counts = ok.sum(axis=1)
    baseline = np.where(
        counts[:, None] >= barrier.MIN_BIN_N,
        touches.sum(axis=1) / np.maximum(counts[:, None], 1),
        np.nan,
    ).T
    actual = barrier.touch_tensor(
        data, feature, horizons, deltas, edges,
        excursions=(low, high), touch_mask=touches,
        baseline_probability=baseline,
    )
    for key in ("conditional_probability", "bin_hit_counts",
                "bin_observation_counts", "eligible_observation_count",
                "bin_assignments"):
        np.testing.assert_allclose(actual[key], expected[key], equal_nan=True)


def test_many_scorer_matches_independent_scoring_across_path_batches():
    frames = [history(n=100, seed=seed) for seed in range(5)]
    paths = np.stack([frame.to_numpy() for frame in frames])
    policies = [
        {"features": None, "bin_edges": None, "feature_name": "roc",
         "params": {"period": period}, "barriers": np.array([-.03, .03]),
         "horizons": np.array([1, 5]), "requested_bin_count": 3}
        for period in (3, 7)
    ]
    expected = [validation.score_histories(paths, **policy) for policy in policies]
    actual = validation.score_histories_many(paths, policies, batch_size=2)
    for wanted, received in zip(expected, actual):
        for field in ("bin_score", "score_supported", "bin_edges"):
            np.testing.assert_allclose(
                getattr(received, field), getattr(wanted, field), rtol=0, atol=1e-10
            )


def test_many_scorer_builds_one_touch_matrix_per_batch_and_horizon(monkeypatch):
    paths = np.stack([history(n=100, seed=seed).to_numpy() for seed in range(5)])
    policies = [
        {"features": None, "bin_edges": None, "feature_name": "roc",
         "params": {"period": period}, "barriers": np.array([-.03, .03]),
         "horizons": np.array([1, 5]), "requested_bin_count": 3}
        for period in (3, 7)
    ]
    original = barrier.barrier_touch_matrix
    calls = []
    def capture(*args):
        calls.append(1)
        return original(*args)
    monkeypatch.setattr(barrier, "barrier_touch_matrix", capture)
    validation.score_histories_many(paths, policies, batch_size=2)
    assert len(calls) == 3 * 2


def test_feature_cache_reuses_rolling_primitives(monkeypatch):
    from barrierlab.domain import torch_features

    paths = torch.as_tensor(history(n=100).to_numpy(copy=True)[None])
    original = torch_features._rolling
    calls = []
    def capture(x, window, op):
        calls.append((window, op))
        return original(x, window, op)
    monkeypatch.setattr(torch_features, "_rolling", capture)
    cache = {}
    torch_features.compute(paths, "ma_ratio", {"period": 10}, cache)
    torch_features.compute(paths, "ma_cross", {"fast": 10, "slow": 50}, cache)
    assert calls.count((10, "mean")) == 1


def test_fixed_external_edges_preserve_collapsed_observed_bins():
    data = history()
    x = pd.Series(np.arange(len(data), dtype=float) % 3)
    deltas, horizons = np.array([-.03, .03]), np.array([7])
    cube = observed_cube(data, x, 10, deltas, horizons)
    actual = validation.score_histories(
        data.to_numpy()[None], x.to_numpy()[None], deltas, horizons, 10,
        bin_edges=cube["bin_edges"],
    )
    np.testing.assert_allclose(actual.bin_score[0], observed_scores(cube), atol=1e-10)


def test_core_feature_recomputation_matches_observed_history():
    from barrierlab.domain import tensor_runtime, torch_features

    data = history()
    paths = tensor_runtime.tensor(data.to_numpy()[None])
    feature = pd.Series(torch_features.compute(paths, "roc", {"period": 5})[0].cpu().numpy())
    deltas, horizons = np.array([-.02, .02, -.06, .06]), np.array([1, 5, 10])
    cube = observed_cube(data, feature, 4, deltas, horizons)
    actual = validation.score_histories(paths, None, deltas, horizons, 4, "roc", {"period": 5})
    np.testing.assert_allclose(actual.bin_score[0], observed_scores(cube), atol=1e-10)


def test_shared_scorer_excludes_thin_bins_even_with_finite_probabilities():
    prob = np.array([[[.2], [.3], [.0]], [[.7], [.6], [1.]]])
    base = np.array([[.3], [.6]])
    scores = scoring.bin_scores(prob, base, np.array([[30], [30], [29]]), [-.1, .1])
    np.testing.assert_allclose(scores.bin_score, [20, 0, 0], atol=1e-12)
    assert scores.score_supported.tolist() == [True, True, False]
    scores = scoring.bin_scores(prob, base, np.array([[30], [29], [29]]), [-.1, .1])
    np.testing.assert_array_equal(scores.bin_score, [0, 0, 0])
    assert not scores.score_supported.any()


def test_unpaired_grid_has_zero_score():
    result = scoring.bin_scores(np.ones((1, 2, 1)), np.ones((1, 1)),
                                np.full((2, 1), 100), [.1])
    np.testing.assert_array_equal(result.bin_score, [0, 0])
    assert not result.score_supported.any()


def test_bin_score_reduces_all_pairs_and_horizons_with_linear_weights():
    # Signed paired shifts: small barrier [10, -20]pp, large [30, -40]pp.
    # Two horizons and a 1/2 small-barrier weight give totals [70, 100]pp.
    prob = np.array([[[.5, .5], [.5, .5]], [[.6, .6], [.3, .3]],
                     [[.5, .5], [.5, .5]], [[.8, .8], [.1, .1]]])
    base = np.full((4, 2), .5)
    counts = np.full((2, 2), 30)
    deltas = np.array([-.05, .05, -.1, .1])
    order = [3, 0, 2, 1]
    result = scoring.bin_scores(prob[order], base[order], counts, deltas[order])
    np.testing.assert_allclose(result.bin_score, [70, 100])
    assert result.score_supported.tolist() == [True, True]
    batch = scoring.bin_scores(np.broadcast_to(prob, (2, 3, *prob.shape)),
                               np.broadcast_to(base, (2, 3, *base.shape)),
                               np.broadcast_to(counts, (2, 3, *counts.shape)), deltas)
    np.testing.assert_allclose(batch.bin_score, np.broadcast_to([70, 100], (2, 3, 2)))


def test_nonfinite_cells_and_empty_horizons_are_unsupported():
    prob = np.full((2, 2, 1), .5)
    prob[0, 0, 0] = np.nan
    result = scoring.bin_scores(prob, np.full((2, 1), .5), np.full((2, 1), 30), [-.1, .1])
    assert not result.score_supported.any()
    result = scoring.bin_scores(prob[..., :0], np.empty((2, 0)), np.empty((2, 0)), [-.1, .1])
    assert result.bin_score.tolist() == [0, 0]
    assert not result.score_supported.any()
