"""Synthetic-OHLC null generation and multiple-testing helpers for Stage 4."""

from __future__ import annotations

import numpy as np
import pandas as pd


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
    shifts = (conditional["prob"][:, :, 0] - baseline[:, None]) * 100.0
    return np.abs(shifts[1] - shifts[0])


def bin_score_from_shift_cube(cube: dict) -> np.ndarray:
    """Sum the Stage 3 cell statistic over each bin's complete shift grid."""
    deltas = np.asarray(cube["Δs"], dtype=float)
    shifts = np.asarray(cube["shift"], dtype=float)
    bin_n = np.asarray(cube["bin_n"], dtype=float)
    magnitudes = sorted({abs(float(value)) for value in deltas if abs(value) > 1e-12})
    paired = [
        magnitude for magnitude in magnitudes
        if np.count_nonzero(np.isclose(deltas, magnitude, atol=1e-12)) == 1
        and np.count_nonzero(np.isclose(deltas, -magnitude, atol=1e-12)) == 1
    ]
    total = np.zeros(shifts.shape[1], dtype=float)
    if not paired:
        return total
    max_magnitude = max(paired)
    for magnitude in paired:
        positive = int(np.flatnonzero(np.isclose(deltas, magnitude, atol=1e-12))[0])
        negative = int(np.flatnonzero(np.isclose(deltas, -magnitude, atol=1e-12))[0])
        skew = np.abs(shifts[positive] - shifts[negative])
        usable = np.isfinite(skew) & np.isfinite(bin_n) & (bin_n > 0)
        # Match selection: a barrier/horizon contributes only when it can
        # compare at least two condition bins.
        usable[:, np.sum(usable, axis=0) < 2] = False
        total += np.where(usable, skew, 0.0).sum(axis=1) * (magnitude / max_magnitude)
    return total


def batched_bin_scores(
    paths, features: np.ndarray | None, delta: float, horizon: int, n_bins: int,
    feature_name: str | None = None, params: dict | None = None,
) -> np.ndarray:
    """Score every bin for every synthetic path in one Torch device batch."""
    import torch

    from barrierlab.domain import tensor_runtime, torch_features

    device = tensor_runtime.device()
    ohlcv = paths if isinstance(paths, torch.Tensor) else tensor_runtime.tensor(paths)
    close, high, low = ohlcv[:, :, 3], ohlcv[:, :, 1], ohlcv[:, :, 2]
    if features is None:
        x = torch_features.compute(ohlcv, feature_name, params or {})
    else:
        x = torch.as_tensor(np.array(features, dtype=float, copy=True), dtype=torch.float64, device=device)
    p, n = close.shape
    valid_n = n - horizon
    lo, hi = torch.full_like(close, float("nan")), torch.full_like(close, float("nan"))
    lo[:, :valid_n] = low[:, 1:].unfold(1, horizon, 1).amin(-1) / close[:, :valid_n] - 1
    hi[:, :valid_n] = high[:, 1:].unfold(1, horizon, 1).amax(-1) / close[:, :valid_n] - 1
    ok = torch.isfinite(x) & torch.isfinite(lo) & torch.isfinite(hi)
    edges = torch.quantile(
        x.nan_to_num(nan=0.0),
        torch.linspace(0.1, 0.9, n_bins - 1, dtype=torch.float64, device=device), dim=1,
    ).T
    idx = (x[:, :, None] >= edges[:, None, :]).sum(-1).long()
    counts = torch.zeros((p, n_bins), dtype=torch.float64, device=device)
    counts.scatter_add_(1, idx, ok.to(torch.float64))
    scores = []
    for touched in (lo <= -abs(delta), hi >= abs(delta)):
        hits = torch.zeros_like(counts)
        hits.scatter_add_(1, idx, (touched & ok).to(torch.float64))
        scores.append(hits / counts.clamp_min(1))
    return (scores[1] - scores[0]).abs().mul(100).cpu().numpy()


def batched_bin_score_sums(
    paths, features: np.ndarray | None, deltas: np.ndarray, horizons: np.ndarray, n_bins: int,
    feature_name: str | None = None, params: dict | None = None, progress=None,
) -> np.ndarray:
    """Return the complete linearly Δ-weighted bin score for every null path."""
    import torch

    from barrierlab.domain import tensor_runtime, torch_features

    device = tensor_runtime.device()
    ohlcv = paths if isinstance(paths, torch.Tensor) else tensor_runtime.tensor(paths)
    close, high, low = ohlcv[:, :, 3], ohlcv[:, :, 1], ohlcv[:, :, 2]
    if features is None:
        x = torch_features.compute(ohlcv, feature_name, params or {})
    else:
        x = torch.as_tensor(np.array(features, dtype=float, copy=True), dtype=torch.float64, device=device)
    path_count, length = close.shape
    edges = torch.quantile(
        x.nan_to_num(nan=0.0),
        torch.linspace(0.1, 0.9, n_bins - 1, dtype=torch.float64, device=device), dim=1,
    ).T
    idx = (x[:, :, None] >= edges[:, None, :]).sum(-1).long()
    deltas = np.asarray(deltas, dtype=float)
    magnitudes = sorted({abs(float(value)) for value in deltas if abs(value) > 1e-12})
    paired = [
        magnitude for magnitude in magnitudes
        if np.count_nonzero(np.isclose(deltas, magnitude, atol=1e-12)) == 1
        and np.count_nonzero(np.isclose(deltas, -magnitude, atol=1e-12)) == 1
    ]
    total = torch.zeros((path_count, n_bins), dtype=torch.float64, device=device)
    if not paired:
        return total.cpu().numpy()
    max_magnitude = max(paired)
    for horizon in np.asarray(horizons, dtype=int):
        try:
            valid_n = length - int(horizon)
            if valid_n <= 0:
                continue
            lo, hi = torch.full_like(close, float("nan")), torch.full_like(close, float("nan"))
            lo[:, :valid_n] = low[:, 1:].unfold(1, int(horizon), 1).amin(-1) / close[:, :valid_n] - 1
            hi[:, :valid_n] = high[:, 1:].unfold(1, int(horizon), 1).amax(-1) / close[:, :valid_n] - 1
            ok = torch.isfinite(x) & torch.isfinite(lo) & torch.isfinite(hi)
            counts = torch.zeros((path_count, n_bins), dtype=torch.float64, device=device)
            counts.scatter_add_(1, idx, ok.to(torch.float64))
            for magnitude in paired:
                scores = []
                for touched in (lo <= -magnitude, hi >= magnitude):
                    hits = torch.zeros_like(counts)
                    hits.scatter_add_(1, idx, (touched & ok).to(torch.float64))
                    scores.append(hits / counts.clamp_min(1))
                total += (scores[1] - scores[0]).abs().mul(100 * magnitude / max_magnitude)
        finally:
            if progress is not None:
                progress.advance()
    return total.cpu().numpy()
