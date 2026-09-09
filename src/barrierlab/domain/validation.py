"""Synthetic-OHLC null generation and multiple-testing helpers for Stage 4."""

from __future__ import annotations

import numpy as np
import pandas as pd

from barrierlab.domain import barrier, scoring


def simulated_ohlc_tensor(data: pd.DataFrame, n_paths: int = 10_000, seed: int = 20260907):
    """Fit and draw the shared OHLC ensemble on the configured Torch device."""
    import torch

    from barrierlab.domain import tensor_runtime

    close = data["close"].to_numpy(float)
    open_ = data["open"].to_numpy(float) if "open" in data else close
    high = data["high"].to_numpy(float) if "high" in data else np.maximum(open_, close)
    low = data["low"].to_numpy(float) if "low" in data else np.minimum(open_, close)
    previous = np.r_[close[0], close[:-1]]
    vectors = np.column_stack((
        np.log(open_ / previous), np.log(close / open_),
        np.log(high / np.maximum(open_, close)), np.log(low / np.minimum(open_, close)),
    ))
    vectors = vectors[np.isfinite(vectors).all(axis=1)]
    device = tensor_runtime.device()
    samples = torch.as_tensor(vectors, dtype=torch.float64, device=device)
    mean, covariance = samples.mean(0), torch.cov(samples.T)
    scale = torch.diagonal(covariance).abs().max().clamp_min(1.0)
    factor = torch.linalg.cholesky(
        covariance + torch.eye(4, dtype=torch.float64, device=device) * scale * 1e-12
    )
    generator = torch.Generator(device=device).manual_seed(seed)
    draws = torch.randn(
        (n_paths, len(data), 4), dtype=torch.float64, device=device, generator=generator
    ) @ factor.T + mean
    initial_close = torch.as_tensor(close[0], dtype=torch.float64, device=device)
    log_open, log_close = draws[:, :, 0], draws[:, :, 1]
    synthetic_close = initial_close * torch.exp(torch.cumsum(log_open + log_close, dim=1))
    previous = torch.cat((initial_close.expand(n_paths, 1), synthetic_close[:, :-1]), dim=1)
    synthetic_open = previous * torch.exp(log_open)
    synthetic_high = torch.maximum(synthetic_open, synthetic_close) * torch.exp(draws[:, :, 2])
    synthetic_low = torch.minimum(synthetic_open, synthetic_close) * torch.exp(draws[:, :, 3])
    volume = torch.as_tensor(
        np.array(data["volume"] if "volume" in data else np.ones(len(data)), dtype=float, copy=True),
        dtype=torch.float64, device=device,
    ).expand(n_paths, -1)
    return torch.stack((synthetic_open, synthetic_high, synthetic_low, synthetic_close, volume), dim=-1)


def baseline_prob(
    data: pd.DataFrame, delta: float, horizon: int,
    excursions: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Two-sided unconditional touch probability for one history."""
    from barrierlab.domain import barrier

    return barrier.touch_tensor(
        data, pd.Series(1.0, index=data.index), np.asarray([horizon]),
        np.asarray([-abs(delta), abs(delta)]), np.empty(0), excursions=excursions,
    )["prob"][:, 0, 0]


def bin_scores(
    data: pd.DataFrame, feature: pd.Series, delta: float, horizon: int, n_bins: int,
    excursions: tuple[np.ndarray, np.ndarray] | None = None,
    baseline: np.ndarray | None = None, edges: np.ndarray | None = None,
) -> np.ndarray:
    """Two-sided skew scores for every condition bin in one history."""
    from barrierlab.domain import barrier

    if baseline is None:
        baseline = baseline_prob(data, delta, horizon, excursions)
    if edges is None:
        edges = barrier.bin_edges(feature, n_bins)
    conditional = barrier.touch_tensor(
        data, feature, np.asarray([horizon]), np.asarray([-abs(delta), abs(delta)]),
        edges, excursions=excursions,
    )
    result = scoring.score_grid(
        conditional["prob"], np.asarray(baseline)[:, None],
        conditional["bin_n"], conditional["Δs"],
    )
    return result["total"].cpu().numpy()


def bin_score_from_shift_cube(cube: dict) -> np.ndarray:
    """Score stored probabilities and baseline with the same kernel as nulls."""
    return scoring.score_grid(
        cube["prob"], cube["base"], cube["bin_n"], cube["Δs"],
    )["total"].cpu().numpy()


def batched_bin_scores(
    paths, features: np.ndarray | None, delta: float, horizon: int, n_bins: int,
    feature_name: str | None = None, params: dict | None = None,
) -> np.ndarray:
    """One-cell adapter to the shared full-grid scoring path."""
    return batched_bin_score_sums(
        paths, features, np.asarray([-abs(delta), abs(delta)]),
        np.asarray([horizon]), n_bins, feature_name, params,
    )


def batched_bin_score_sums(
    paths, features: np.ndarray | None, deltas: np.ndarray, horizons: np.ndarray, n_bins: int,
    feature_name: str | None = None, params: dict | None = None, progress=None,
    *, edges: np.ndarray | None = None,
) -> np.ndarray:
    """Measure each history and call the common baseline-relative grid scorer.

    n_bins is the requested quantile count, not the count after ties collapse.
    Collapsed paths have trailing zero-score bins. Fixed external features may
    supply the observed edges instead. Horizons stream through the scorer to
    avoid holding an entire simulated probability cube in device memory.
    """
    import torch

    from barrierlab.domain import tensor_runtime, torch_features

    ohlcv = paths if isinstance(paths, torch.Tensor) else tensor_runtime.tensor(paths)
    close, high, low = ohlcv[:, :, 3], ohlcv[:, :, 1], ohlcv[:, :, 2]
    x = (torch_features.compute(ohlcv, feature_name, params or {}) if features is None
         else tensor_runtime.tensor(features))
    if x.shape != close.shape:
        raise ValueError("features must match the path and time axes")
    path_count, length = close.shape
    if edges is None:
        edges_t = barrier.batched_bin_edges(x, n_bins)
    else:
        edges_t = tensor_runtime.tensor(edges).expand(path_count, -1).contiguous()
        n_bins = edges_t.shape[1] + 1
    indices = barrier.bin_indices(x, edges_t)
    baseline_feature = torch.ones_like(x)
    baseline_indices = torch.zeros_like(indices)
    deltas = np.asarray(deltas, dtype=float)
    total = close.new_zeros((path_count, n_bins))
    for horizon in np.asarray(horizons, dtype=int):
        try:
            if horizon < 1:
                raise ValueError("horizons must be positive")
            valid_n = length - int(horizon)
            if valid_n <= 0 or not len(deltas):
                continue
            lo, hi = torch.full_like(close, float("nan")), torch.full_like(close, float("nan"))
            lo[:, :valid_n] = low[:, 1:].unfold(1, int(horizon), 1).amin(-1) / close[:, :valid_n] - 1
            hi[:, :valid_n] = high[:, 1:].unfold(1, int(horizon), 1).amax(-1) / close[:, :valid_n] - 1
            probabilities, baselines = [], []
            for delta in deltas:
                prob, _, counts, _ = barrier.touch_rates(lo, hi, x, indices, n_bins, delta)
                base, _, _, _ = barrier.touch_rates(
                    lo, hi, baseline_feature, baseline_indices, 1, delta,
                )
                probabilities.append(prob)
                baselines.append(base[:, 0])
            result = scoring.score_grid(
                torch.stack(probabilities, dim=1).unsqueeze(-1),
                torch.stack(baselines, dim=1).unsqueeze(-1),
                counts.unsqueeze(-1), deltas,
            )
            total += result["total"]
        finally:
            if progress is not None:
                progress.advance()
    return total.cpu().numpy()
