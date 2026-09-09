"""Synthetic-OHLC null generation and full-bin measurement for Stage 3."""

from __future__ import annotations

import numpy as np
import pandas as pd

from barrierlab.domain import barrier, scoring


def simulated_ohlc_tensor(data: pd.DataFrame, n_paths: int = 1_000, seed: int = 20260907):
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
    *, edges: np.ndarray | None = None, touches=None, baseline=None, bin_indices=None,
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
        touches=touches, baselines=baseline,
        bin_indices_cache=bin_indices,
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


HISTORY_BATCH_SIZE = 256


def _score_histories_many_batch(paths, policies: list[dict]) -> list[dict[str, np.ndarray]]:
    """Score several conditions while traversing shared price outcomes once.

    Feature calculation, quantiles, and conditional counts remain specific to
    each condition. Forward extremes and unconditional barrier rates depend
    only on the market histories, so Stage 3 computes those once per group.
    """
    import torch

    from barrierlab.domain import tensor_runtime, torch_features

    if not policies:
        return []
    ohlcv = paths.to(dtype=torch.float64) if isinstance(paths, torch.Tensor) else tensor_runtime.tensor(paths)
    prepared = []
    feature_cache = {}
    for policy in policies:
        features = policy.get("features")
        x = (torch_features.compute(
            ohlcv, policy.get("feature_name"), policy.get("params") or {}, feature_cache,
        )
             if features is None else tensor_runtime.tensor(features))
        x = x.expand(ohlcv.shape[:2])
        supplied_edges = policy.get("edges")
        if supplied_edges is None:
            edges = barrier.batched_bin_edges(x, policy["n_bins"])
        else:
            edges = tensor_runtime.tensor(supplied_edges).expand(ohlcv.shape[0], -1).contiguous()
        prepared.append({
            "x": x, "edges_t": edges,
            "indices": barrier.bin_indices(x, edges),
            "scores": ohlcv.new_zeros((ohlcv.shape[0], edges.shape[1] + 1)),
            "valid": torch.zeros((ohlcv.shape[0], edges.shape[1] + 1), dtype=torch.bool,
                                 device=ohlcv.device),
        })

    deltas = np.asarray(policies[0]["deltas"], dtype=float)
    horizons = np.asarray(policies[0]["horizons"], dtype=int)
    if any(not np.array_equal(deltas, np.asarray(policy["deltas"], dtype=float))
           or not np.array_equal(horizons, np.asarray(policy["horizons"], dtype=int))
           for policy in policies[1:]):
        raise ValueError("shared scoring policies must use identical barrier and horizon grids")

    for horizon_index, (lo, hi) in enumerate(barrier.iter_extremes(ohlcv, horizons)):
        touched, price_ok = barrier.barrier_touch_matrix(lo, hi, deltas)
        base = barrier.reduce_baseline_touch_matrix(touched, price_ok, ohlcv.dtype)[0].unsqueeze(-1)
        for item in prepared:
            prob, _, counts, _ = barrier.reduce_touch_matrix(
                touched, price_ok, item["x"], item["indices"], item["scores"].shape[1],
            )
            prob = prob.unsqueeze(-1)
            result = scoring.bin_scores(prob, base, counts.unsqueeze(-1), deltas)
            item["scores"] += result["scores"]
            item["valid"] |= result["valid"]

    return [
        {"scores": item["scores"].cpu().numpy(), "valid": item["valid"].cpu().numpy(),
         "edges": item["edges_t"].cpu().numpy()}
        for item in prepared
    ]


def score_histories_many(
    paths, policies: list[dict], *, batch_size: int = HISTORY_BATCH_SIZE, progress=None,
) -> list[dict[str, np.ndarray]]:
    """Score conditions together in bounded batches of market histories."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not policies:
        return []
    deltas = np.asarray(policies[0]["deltas"], dtype=float)
    horizons = np.asarray(policies[0]["horizons"], dtype=int)
    if any(not np.array_equal(deltas, np.asarray(policy["deltas"], dtype=float))
           or not np.array_equal(horizons, np.asarray(policy["horizons"], dtype=int))
           for policy in policies[1:]):
        raise ValueError("shared scoring policies must use identical barrier and horizon grids")
    parts = [[] for _ in policies]
    for start in range(0, len(paths), batch_size):
        batch = _score_histories_many_batch(paths[start:start + batch_size], policies)
        for destination, result in zip(parts, batch):
            destination.append(result)
        if progress is not None:
            progress.advance()
    return [
        {key: np.concatenate([part[key] for part in results], axis=0)
         for key in ("scores", "valid", "edges")}
        for results in parts
    ]
