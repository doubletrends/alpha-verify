"""Numerical parity across measurement, bin scoring, and null validation."""

import numpy as np
import pandas as pd
import pytest
import torch

from alphaverify.domain import barrier, scoring, shift, validation
from alphaverify.domain.notation import MIN_BIN_N


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
    return {**cube, "probability_shift": shift.from_cube(cube, baseline),
            "baseline_probability": baseline}


def observed_scores(cube):
    return scoring.bin_scores(
        cube["conditional_probability"], cube["baseline_probability"],
        cube["bin_observation_counts"], cube["barriers"],
    ).bin_score.cpu().numpy()


@pytest.mark.parametrize("kind,n_bins", [("continuous", 10),
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


def test_measurement_baseline_includes_feature_warmup_and_uses_float64(monkeypatch):
    monkeypatch.setattr(barrier, "MIN_BIN_N", 1)
    data = pd.DataFrame({"open": [100.] * 5, "high": [100., 101., 110., 103., 120.],
                         "low": [100., 99., 90., 98., 95.], "close": [100.] * 5,
                         "volume": [1.] * 5})
    features = np.array([[np.nan, np.nan, 0., 1., 1.]])
    edges, slices = barrier.measure_histories(barrier.ohlcv_tensor(data).float(), features,
                                              [-.05, .05], [1], 2, bin_edges=[.5])
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


def test_persistable_observed_cache_matches_uncached_measurement():
    data = history(n=100)
    data.loc[15, "high"] = np.nan
    feature = pd.Series(np.sin(np.arange(100)))
    # Unsorted/repeated horizons exercise cache lookup.
    deltas, horizons = np.array([-.03, 0., .03]), np.array([7, 1, 7, 3])
    edges = barrier.bin_edges(feature, 3)
    expected = barrier.touch_tensor(data, feature, horizons, deltas, edges)
    outcomes = barrier.observed_outcomes(data, deltas, horizons)
    lo = outcomes["downside_excursion"][horizons - 1]
    hi = outcomes["upside_excursion"][horizons - 1]
    delta_matrix = deltas.reshape(1, 1, -1)
    ok = np.isfinite(lo) & np.isfinite(hi)
    touches = np.where(
        delta_matrix < 0, lo[:, :, None] <= delta_matrix, hi[:, :, None] >= delta_matrix,
    ) & ok[:, :, None]
    counts = ok.sum(axis=1)
    baseline = np.where(
        counts[:, None] >= MIN_BIN_N,
        touches.sum(axis=1) / np.maximum(counts[:, None], 1),
        np.nan,
    ).T
    np.testing.assert_array_equal(outcomes["touch_mask"], touches)
    np.testing.assert_array_equal(outcomes["baseline_probability"], baseline)
    actual = barrier.touch_tensor(
        data, feature, horizons, deltas, edges,
        excursions=(outcomes["downside_excursion"], outcomes["upside_excursion"]),
        touch_mask=outcomes["touch_mask"],
        baseline_probability=outcomes["baseline_probability"],
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


def test_simulated_ensemble_is_valid_ohlcv_and_reproducible_from_its_seed():
    data = history(n=200)
    ensemble = validation.simulated_ohlc_tensor(data, 6, 20260907)
    open_, high, low, close, volume = (ensemble[:, :, i] for i in range(5))
    assert torch.all(high >= torch.maximum(open_, close)) and torch.all(low <= torch.minimum(open_, close))
    np.testing.assert_array_equal(volume.numpy(), np.broadcast_to(data["volume"].to_numpy(), (6, 200)))
    np.testing.assert_array_equal(ensemble.numpy(), validation.simulated_ohlc_tensor(data, 6, 20260907).numpy())


def test_bin_score_reduces_all_pairs_and_horizons_with_linear_weights():
    # Signed paired shifts: small barrier [.10, -.20], large [.30, -.40].
    # Two horizons and a 1/2 small-barrier weight give totals [.70, 1.00].
    prob = np.array([[[.5, .5], [.5, .5]], [[.6, .6], [.3, .3]],
                     [[.5, .5], [.5, .5]], [[.8, .8], [.1, .1]]])
    base = np.full((4, 2), .5)
    counts = np.full((2, 2), 30)
    deltas = np.array([-.05, .05, -.1, .1])
    order = [3, 0, 2, 1]
    result = scoring.bin_scores(prob[order], base[order], counts, deltas[order])
    np.testing.assert_allclose(result.bin_score, [.70, 1.00])
    assert result.score_supported.tolist() == [True, True]


def test_full_grid_score_matches_pair_reference_with_partial_support():
    rng = np.random.default_rng(24)
    # Unsorted grid, zero, an unmatched row, and duplicate rows.
    barriers = np.array([.02, -.01, 0., -.02, .01, .03, -.04, .04, .04])
    probability = rng.uniform(.1, .9, (3, len(barriers), 4, 7))
    baseline = rng.uniform(.1, .9, (3, len(barriers), 7))
    counts = rng.integers(25, 45, (3, 4, 7))
    probability[0, 0, 0, 0] = np.nan
    probability[1, 1, 2, 3] = np.inf
    baseline[2, 4, 5] = np.nan
    counts[0, :, 6] = 0
    shifts = probability - baseline[:, :, None, :]
    expected = np.zeros((3, 4))
    supported = np.zeros((3, 4), dtype=bool)
    for path in range(3):
        for horizon in range(7):
            for positive, negative, weight in [(4, 1, .5), (0, 3, 1.)]:
                difference = shifts[path, positive, :, horizon] - shifts[path, negative, :, horizon]
                valid = np.isfinite(difference) & (counts[path, :, horizon] >= 30)
                if valid.sum() >= 2:
                    expected[path, valid] += weight * np.abs(difference[valid])
                    supported[path, valid] = True
    result = scoring.bin_scores(probability, baseline, counts, barriers)
    np.testing.assert_allclose(result.bin_score, expected, atol=1e-10)
    np.testing.assert_array_equal(result.score_supported, supported)
