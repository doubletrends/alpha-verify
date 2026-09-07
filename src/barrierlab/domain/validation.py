"""Synthetic-OHLC null generation and multiple-testing helpers for Stage 4."""

from __future__ import annotations

import numpy as np
import pandas as pd


def simulated_ohlc_tensor(data: pd.DataFrame, n_paths: int = 1000, seed: int = 20260907):
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


def bh(p_values: np.ndarray, q: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg FDR correction, preserving input order."""
    p = np.asarray(p_values, dtype=float)
    finite = np.isfinite(p)
    rejected, qvals = np.zeros(p.shape, dtype=bool), np.full(p.shape, np.nan)
    m = int(finite.sum())
    if m == 0:
        return rejected, qvals
    order = np.argsort(p[finite], kind="stable")
    ps, ranks = p[finite][order], np.arange(1, m + 1)
    below = ps <= ranks * q / m
    rejected_sorted = np.zeros(m, dtype=bool)
    if below.any():
        rejected_sorted[:int(np.flatnonzero(below).max()) + 1] = True
    q_sorted = np.clip(np.minimum.accumulate((ps * m / ranks)[::-1])[::-1], 0.0, 1.0)
    inverse = np.empty(m, dtype=int)
    inverse[order] = np.arange(m)
    positions = np.flatnonzero(finite)
    rejected[positions], qvals[positions] = rejected_sorted[inverse], q_sorted[inverse]
    return rejected, qvals
