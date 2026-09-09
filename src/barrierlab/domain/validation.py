"""Synthetic-OHLC null generation and full-bin measurement for Stage 3."""

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


def score_histories(
    paths, features: np.ndarray | None, deltas: np.ndarray, horizons: np.ndarray, n_bins: int,
    feature_name: str | None = None, params: dict | None = None, progress=None,
    *, edges: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Common observed/null calculation, with no distinction based on role.

    Core features are recomputed when features is None; fixed observed feature
    values and edges can instead be shared across all paths. Measurement owns
    quantiles, counts, baselines, and horizon streaming. Return scores/validity
    plus the actual edges, so observed labels follow the recomputed bins.
    """
    import torch

    from barrierlab.domain import tensor_runtime, torch_features

    ohlcv = paths.to(dtype=torch.float64) if isinstance(paths, torch.Tensor) else tensor_runtime.tensor(paths)
    x = torch_features.compute(ohlcv, feature_name, params or {}) if features is None else features
    edges_t, measurements = barrier.measure_histories(
        ohlcv, x, deltas, horizons, n_bins, edges=edges,
    )
    total = ohlcv.new_zeros((ohlcv.shape[0], edges_t.shape[1] + 1))
    valid = torch.zeros_like(total, dtype=torch.bool)
    for measured in measurements:
        result = scoring.bin_scores(measured["prob"], measured["base"], measured["bin_n"], deltas)
        total += result["scores"]
        valid |= result["valid"]
        if progress is not None:
            progress.advance()
    return {"scores": total.cpu().numpy(), "valid": valid.cpu().numpy(),
            "edges": edges_t.cpu().numpy()}
