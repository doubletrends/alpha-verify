"""Synthetic-OHLC null generation and full-bin measurement for Stage 3."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from alphaverify.domain import barrier, scoring, tensor_runtime, torch_features
from alphaverify.domain.notation import HistoryScoreResult, OhlcvComponent

REPLICATE_BATCH_SIZE = 256
# Peak scoring memory per bar per replicate, measured on float64 CUDA kernels and rounded up:
# a fixed share plus a share per scored condition and per signed barrier.
_BYTES_PER_BAR = 80
_BYTES_PER_BAR_PER_CONDITION = 25
_BYTES_PER_BAR_PER_BARRIER = 10


def replicate_batch_size(n_bars: int, n_conditions: int, n_barriers: int, memory_budget: int) -> int:
    """Replicates to score together: the most that fit the budget, between 1 and REPLICATE_BATCH_SIZE.

    Scores do not depend on the batch size, so this only bounds memory.
    """
    per_replicate = n_bars * (
        _BYTES_PER_BAR
        + _BYTES_PER_BAR_PER_CONDITION * n_conditions
        + _BYTES_PER_BAR_PER_BARRIER * n_barriers
    )
    return int(max(1, min(REPLICATE_BATCH_SIZE, memory_budget // max(per_replicate, 1))))


def simulated_ohlc_tensor(data: pd.DataFrame, n_replicates: int, seed: int):
    """Fit and draw the shared OHLC ensemble on the configured device; return it in host memory.

    The draw is one call, because chunked draws do not reproduce the same stream.
    Each finished series moves to host memory before the next is computed, so the
    device holds the drawn components and at most a few series at a time. Scoring
    moves one replicate batch back to the device.
    """
    ohlcv = barrier.ohlcv_array(data)
    open_, high, low, close, observed_volume = (
        ohlcv[:, component] for component in OhlcvComponent
    )
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
    # CPU and CUDA generators yield different streams for one seed, and the device is
    # not part of the Stage 3 method record: results are reproducible per device only.
    generator = torch.Generator(device=device).manual_seed(seed)
    synthetic_components = torch.randn(
        (n_replicates, len(data), 4),
        dtype=torch.float64, device=device, generator=generator,
    ) @ cholesky_factor.T
    synthetic_components += component_mean
    ensemble = torch.empty((n_replicates, len(data), 5), dtype=torch.float64)
    initial_close = torch.as_tensor(close[0], dtype=torch.float64, device=device)
    log_open = synthetic_components[:, :, 0]
    log_close = synthetic_components[:, :, 1]
    synthetic_close = initial_close * torch.exp(torch.cumsum(log_open + log_close, dim=1))
    previous = torch.cat(
        (initial_close.expand(n_replicates, 1), synthetic_close[:, :-1]), dim=1
    )
    synthetic_open = previous * torch.exp(log_open)
    del previous
    ensemble[:, :, OhlcvComponent.HIGH] = (
        torch.maximum(synthetic_open, synthetic_close) * torch.exp(synthetic_components[:, :, 2])
    ).cpu()
    ensemble[:, :, OhlcvComponent.LOW] = (
        torch.minimum(synthetic_open, synthetic_close) * torch.exp(synthetic_components[:, :, 3])
    ).cpu()
    del synthetic_components, log_open, log_close
    ensemble[:, :, OhlcvComponent.OPEN] = synthetic_open.cpu()
    ensemble[:, :, OhlcvComponent.CLOSE] = synthetic_close.cpu()
    ensemble[:, :, OhlcvComponent.VOLUME] = torch.as_tensor(observed_volume, dtype=torch.float64)
    return ensemble


def score_histories(
    paths, features: np.ndarray | None, barriers: np.ndarray, horizons: np.ndarray,
    requested_bin_count: int,
    feature_name: str | None = None, params: dict | None = None,
    *, bin_edges: np.ndarray | None = None, touch_mask=None,
    baseline_probability=None, bin_assignments=None,
) -> HistoryScoreResult:
    """Common observed/null calculation, with no distinction based on role.

    Core features are recomputed when features is None; fixed observed feature
    values and edges can instead be shared across all paths. Measurement owns
    quantiles, counts, baselines, and horizon streaming. Return scores/validity
    plus the actual edges, so observed labels follow the recomputed bins.
    """
    ohlcv = tensor_runtime.tensor(paths)
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
    return HistoryScoreResult(
        bin_score=bin_score.cpu().numpy(),
        score_supported=score_supported.cpu().numpy(),
        bin_edges=edges_t.cpu().numpy(),
    )


def _score_histories_many_batch(paths, policies: list[dict]) -> list[HistoryScoreResult]:
    """Score several conditions while traversing shared price outcomes once.

    Feature calculation, quantiles, and conditional counts remain specific to
    each condition. Forward extremes and unconditional barrier rates depend
    only on the market histories, so Stage 3 computes those once per group.
    Callers validate that all policies share one barrier/horizon grid.
    """
    if not policies:
        return []
    ohlcv = tensor_runtime.tensor(paths)
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

    for downside_excursion, upside_excursion in barrier.iter_extremes(ohlcv, horizons):
        touch_mask, price_eligible = barrier.barrier_touch_matrix(
            downside_excursion, upside_excursion, barriers
        )
        baseline_probability = barrier.reduce_baseline_touch_matrix(
            touch_mask, price_eligible
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
    paths, policies: list[dict], *, batch_size: int, progress=None,
) -> list[HistoryScoreResult]:
    """Score conditions together in batches of ``batch_size`` market histories.

    ``replicate_batch_size`` sizes batches to a memory budget; scores do not depend on it.
    """
    if not policies:
        return []
    barriers = np.asarray(policies[0]["barriers"], dtype=float)
    horizons = np.asarray(policies[0]["horizons"], dtype=int)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
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
