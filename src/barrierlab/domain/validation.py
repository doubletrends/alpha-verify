"""Synthetic-OHLC null generation and full-bin measurement for Stage 3."""

from __future__ import annotations

import numpy as np
import pandas as pd

from barrierlab.domain import barrier, scoring
from barrierlab.domain.notation import HistoryScoreResult


def simulated_ohlc_tensor(
    data: pd.DataFrame, n_replicates: int = 1_000, seed: int = 20260907
):
    """Fit and draw the shared OHLC ensemble on the configured Torch device."""
    import torch

    from barrierlab.domain import tensor_runtime

    close = data["close"].to_numpy(float)
    open_ = data["open"].to_numpy(float) if "open" in data else close
    high = data["high"].to_numpy(float) if "high" in data else np.maximum(open_, close)
    low = data["low"].to_numpy(float) if "low" in data else np.minimum(open_, close)
    previous = np.r_[close[0], close[:-1]]
    log_ohlc_components = np.column_stack((
        np.log(open_ / previous), np.log(close / open_),
        np.log(high / np.maximum(open_, close)), np.log(low / np.minimum(open_, close)),
    ))
    log_ohlc_components = log_ohlc_components[
        np.isfinite(log_ohlc_components).all(axis=1)
    ]
    device = tensor_runtime.device()
    samples = torch.as_tensor(
        log_ohlc_components, dtype=torch.float64, device=device
    )
    component_mean, component_covariance = samples.mean(0), torch.cov(samples.T)
    jitter_scale = torch.diagonal(component_covariance).abs().max().clamp_min(1.0)
    cholesky_factor = torch.linalg.cholesky(
        component_covariance
        + torch.eye(4, dtype=torch.float64, device=device) * jitter_scale * 1e-12
    )
    generator = torch.Generator(device=device).manual_seed(seed)
    synthetic_components = torch.randn(
        (n_replicates, len(data), 4),
        dtype=torch.float64, device=device, generator=generator,
    ) @ cholesky_factor.T + component_mean
    initial_close = torch.as_tensor(close[0], dtype=torch.float64, device=device)
    log_open = synthetic_components[:, :, 0]
    log_close = synthetic_components[:, :, 1]
    synthetic_close = initial_close * torch.exp(torch.cumsum(log_open + log_close, dim=1))
    previous = torch.cat(
        (initial_close.expand(n_replicates, 1), synthetic_close[:, :-1]), dim=1
    )
    synthetic_open = previous * torch.exp(log_open)
    synthetic_high = torch.maximum(synthetic_open, synthetic_close) * torch.exp(
        synthetic_components[:, :, 2]
    )
    synthetic_low = torch.minimum(synthetic_open, synthetic_close) * torch.exp(
        synthetic_components[:, :, 3]
    )
    volume = torch.as_tensor(
        np.array(data["volume"] if "volume" in data else np.ones(len(data)), dtype=float, copy=True),
        dtype=torch.float64, device=device,
    ).expand(n_replicates, -1)
    return torch.stack((synthetic_open, synthetic_high, synthetic_low, synthetic_close, volume), dim=-1)


def score_histories(
    paths, features: np.ndarray | None, barriers: np.ndarray, horizons: np.ndarray,
    requested_bin_count: int,
    feature_name: str | None = None, params: dict | None = None, progress=None,
    *, bin_edges: np.ndarray | None = None, touch_mask=None,
    baseline_probability=None, bin_assignments=None,
) -> HistoryScoreResult:
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
        ohlcv, x, barriers, horizons, requested_bin_count, bin_edges=bin_edges,
        touch_mask=touch_mask, baseline_probability=baseline_probability,
        bin_assignments=bin_assignments,
    )
    bin_score = ohlcv.new_zeros((ohlcv.shape[0], edges_t.shape[1] + 1))
    score_supported = torch.zeros_like(bin_score, dtype=torch.bool)
    for measured in measurements:
        result = scoring.bin_scores(
            measured.conditional_probability,
            measured.baseline_probability,
            measured.bin_observation_counts,
            barriers,
        )
        bin_score += result.bin_score
        score_supported |= result.score_supported
        if progress is not None:
            progress.advance()
    return HistoryScoreResult(
        bin_score=bin_score.cpu().numpy(),
        score_supported=score_supported.cpu().numpy(),
        bin_edges=edges_t.cpu().numpy(),
    )


REPLICATE_BATCH_SIZE = 256


def _score_histories_many_batch(paths, policies: list[dict]) -> list[HistoryScoreResult]:
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
        supplied_edges = policy.get("bin_edges")
        if supplied_edges is None:
            edges = barrier.batched_bin_edges(x, policy["requested_bin_count"])
        else:
            edges = tensor_runtime.tensor(supplied_edges).expand(ohlcv.shape[0], -1).contiguous()
        prepared.append({
            "feature_values": x, "bin_edges": edges,
            "bin_assignments": barrier.bin_indices(x, edges),
            "bin_score": ohlcv.new_zeros((ohlcv.shape[0], edges.shape[1] + 1)),
            "score_supported": torch.zeros((ohlcv.shape[0], edges.shape[1] + 1), dtype=torch.bool,
                                 device=ohlcv.device),
        })

    barriers = np.asarray(policies[0]["barriers"], dtype=float)
    horizons = np.asarray(policies[0]["horizons"], dtype=int)
    if any(not np.array_equal(barriers, np.asarray(policy["barriers"], dtype=float))
           or not np.array_equal(horizons, np.asarray(policy["horizons"], dtype=int))
           for policy in policies[1:]):
        raise ValueError("shared scoring policies must use identical barrier and horizon grids")

    for downside_excursion, upside_excursion in barrier.iter_extremes(ohlcv, horizons):
        touch_mask, price_eligible = barrier.barrier_touch_matrix(
            downside_excursion, upside_excursion, barriers
        )
        baseline_probability = barrier.reduce_baseline_touch_matrix(
            touch_mask, price_eligible, ohlcv.dtype
        )[0].unsqueeze(-1)
        for item in prepared:
            conditional_probability, _, bin_observation_counts, _ = barrier.reduce_touch_matrix(
                touch_mask, price_eligible, item["feature_values"],
                item["bin_assignments"], item["bin_score"].shape[1],
            )
            result = scoring.bin_scores(
                conditional_probability.unsqueeze(-1), baseline_probability,
                bin_observation_counts.unsqueeze(-1), barriers,
            )
            item["bin_score"] += result.bin_score
            item["score_supported"] |= result.score_supported

    return [
        HistoryScoreResult(
            bin_score=item["bin_score"].cpu().numpy(),
            score_supported=item["score_supported"].cpu().numpy(),
            bin_edges=item["bin_edges"].cpu().numpy(),
        )
        for item in prepared
    ]


def score_histories_many(
    paths, policies: list[dict], *, batch_size: int = REPLICATE_BATCH_SIZE, progress=None,
) -> list[HistoryScoreResult]:
    """Score conditions together in bounded batches of market histories."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not policies:
        return []
    barriers = np.asarray(policies[0]["barriers"], dtype=float)
    horizons = np.asarray(policies[0]["horizons"], dtype=int)
    if any(not np.array_equal(barriers, np.asarray(policy["barriers"], dtype=float))
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
    return [HistoryScoreResult(
        bin_score=np.concatenate([part.bin_score for part in results], axis=0),
        score_supported=np.concatenate(
            [part.score_supported for part in results], axis=0
        ),
        bin_edges=np.concatenate([part.bin_edges for part in results], axis=0),
    ) for results in parts]
