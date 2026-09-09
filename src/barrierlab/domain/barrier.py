"""
Barrier-touch surfaces: will price reach ±Δ within t bars, given a condition?

This is the engine's single measurement: the probability that price reaches a barrier,
given a condition. Everything the pipeline used to compute as a separate "outcome" is a
row of it -- the -10% drawdown surface is the Δ = -10% row, the +10% runup surface
is the +10% row, and the asymmetry between them is the two rows read against each other
in the same column. There is no outcome registry any more; Δ *is* the axis.

Two things differ from the close-to-close machinery this replaces:

  Intraday extremes.  A barrier is touched when the bar's low or high reaches it, not
  when the close does. Measured on BTC daily, close-only understates a -10%/14d touch
  at 21.7% against 28.7% on the lows, and a -5%/7d touch at 28.6% against 39.5%. For a
  question whose whole point is where to put a stop, closes are the wrong series.

  Quantile bins.  Conditions are the feature's deciles rather than equal-width slices
  of its 2nd-98th percentile range. Equal-width bins put almost no observations in the
  tails, which is exactly where the interesting conditions live; deciles guarantee an
  equal, known sample behind every column and make each column a condition you could
  actually trade ("the feature is in its bottom tenth").
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from barrierlab.domain import tensor_runtime

# Below this many observations a bin's rate is not worth reporting.
MIN_BIN_N = 30
MEASUREMENT_VERSION = "shared-outcome-cache-float64-v2"


def configure_cuda(enabled: bool) -> None:
    """Select the unified Torch numerical device for one CLI invocation."""
    tensor_runtime.configure(enabled)


def bin_edges(feature: pd.Series, n_bins: int = 10) -> np.ndarray:
    """
    Interior quantile edges of the feature, so each bin holds ~1/n_bins of the sample.

    Returns at most n_bins-1 edges. A feature with heavy ties -- an integer calendar
    feature, or the constant used by the baseline node -- yields duplicate quantiles,
    so the caller may get fewer bins than asked for. That is correct, not an error.

    Assignment is searchsorted-left: values <= e enter the lower bin. Retain
    the historical edge filter min(v) < e <= max(v), which eliminates phantom
    edges for constant features. A maximum-valued edge can leave an empty final
    bin; touch probabilities and scoring exclude it through sample counts.
    """
    values = tensor_runtime.tensor(feature.to_numpy(float))[None, :]
    edges = batched_bin_edges(values, n_bins)[0]
    return edges[torch.isfinite(edges)].cpu().numpy()


def batched_bin_edges(values: torch.Tensor, n_bins: int) -> torch.Tensor:
    """Finite-value quantiles, deduplicated per path and padded with infinity."""
    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    if n_bins == 1:
        return values.new_empty((values.shape[0], 0))
    finite = torch.isfinite(values)
    clean = torch.where(finite, values, float("nan"))
    qs = torch.linspace(0, 1, n_bins + 1, dtype=values.dtype, device=values.device)[1:-1]
    edges = torch.nanquantile(clean, qs, dim=1).T
    lo = torch.where(finite, values, float("inf")).amin(dim=1, keepdim=True)
    hi = torch.where(finite, values, float("-inf")).amax(dim=1, keepdim=True)
    unique = torch.ones_like(edges, dtype=torch.bool)
    unique[:, 1:] = edges[:, 1:] != edges[:, :-1]
    valid = torch.isfinite(edges) & unique & (edges > lo) & (edges <= hi)
    return torch.where(valid, edges, float("inf")).sort(dim=1).values.contiguous()


def bin_indices(values: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    """The common left-boundary convention, including exact ties."""
    return torch.searchsorted(edges.contiguous(), values.contiguous(), right=False)


def touch_rates(lo, hi, feature, indices, n_bins, delta, min_n=MIN_BIN_N):
    """Batched conditional rates and counts for one signed barrier/horizon.

    Inputs have (history, time) axes. The baseline is measured independently
    using a constant feature, so it includes all eligible market dates.
    """
    ok = torch.isfinite(lo) & torch.isfinite(hi) & torch.isfinite(feature)
    counts = lo.new_zeros((lo.shape[0], n_bins))
    counts.scatter_add_(1, indices, ok.to(lo.dtype))
    touched = lo <= delta if delta < 0 else hi >= delta
    hits = torch.zeros_like(counts)
    hits.scatter_add_(1, indices, (touched & ok).to(lo.dtype))
    prob = torch.where(counts >= min_n, hits / counts.clamp_min(1), float("nan"))
    return prob, hits, counts, ok.sum(dim=1)


def touch_rate_matrix(lo, hi, feature, indices, n_bins, deltas, min_n=MIN_BIN_N):
    """Conditional rates for every signed barrier in one tensor operation.

    Inputs have ``(history, time)`` axes. Results have
    ``(history, signed barrier, bin)`` axes; counts do not have a barrier axis.
    """
    touched, price_ok = barrier_touch_matrix(lo, hi, deltas)
    return reduce_touch_matrix(touched, price_ok, feature, indices, n_bins, min_n)


def barrier_touch_matrix(lo, hi, deltas):
    """Return the shared ``history × time × delta`` price-touch matrix."""
    deltas_t = torch.as_tensor(deltas, dtype=lo.dtype, device=lo.device)
    if deltas_t.ndim != 1:
        raise ValueError("barriers must be a vector")
    price_ok = torch.isfinite(lo) & torch.isfinite(hi)
    if not len(deltas_t):
        return torch.empty((*lo.shape, 0), dtype=torch.bool, device=lo.device), price_ok
    delta_matrix = deltas_t.view(1, 1, -1)
    touched = torch.where(
        delta_matrix < 0,
        lo.unsqueeze(-1) <= delta_matrix,
        hi.unsqueeze(-1) >= delta_matrix,
    ) & price_ok.unsqueeze(-1)
    return touched, price_ok


def reduce_touch_matrix(touched, price_ok, feature, indices, n_bins, min_n=MIN_BIN_N):
    """Reduce one shared touch matrix through a condition's bin assignment."""
    ok = price_ok & torch.isfinite(feature)
    counts = feature.new_zeros((feature.shape[0], n_bins))
    counts.scatter_add_(1, indices, ok.to(feature.dtype))
    if touched.shape[-1] == 0:
        empty = feature.new_empty((feature.shape[0], 0, n_bins))
        return empty, empty.clone(), counts, ok.sum(dim=1)
    hits = feature.new_zeros((feature.shape[0], n_bins, touched.shape[-1]))
    hits.scatter_add_(
        1,
        indices.unsqueeze(-1).expand(-1, -1, touched.shape[-1]),
        (touched & ok.unsqueeze(-1)).to(feature.dtype),
    )
    probabilities = torch.where(
        counts.unsqueeze(-1) >= min_n,
        hits / counts.unsqueeze(-1).clamp_min(1),
        float("nan"),
    )
    return probabilities.transpose(1, 2), hits.transpose(1, 2), counts, ok.sum(dim=1)


def baseline_rate_matrix(lo, hi, deltas, min_n=MIN_BIN_N):
    """Unconditional rates for every signed barrier, without bin scattering."""
    touched, ok = barrier_touch_matrix(lo, hi, deltas)
    return reduce_baseline_touch_matrix(touched, ok, lo.dtype, min_n)


def reduce_baseline_touch_matrix(touched, price_ok, dtype=torch.float64, min_n=MIN_BIN_N):
    """Reduce a previously generated shared touch matrix to baseline rates."""
    counts = price_ok.sum(dim=1).to(dtype)
    if touched.shape[-1] == 0:
        return torch.empty((touched.shape[0], 0), dtype=dtype, device=touched.device), counts
    hits = touched.sum(dim=1).to(dtype)
    return torch.where(
        counts.unsqueeze(-1) >= min_n,
        hits / counts.unsqueeze(-1).clamp_min(1),
        float("nan"),
    ), counts


def bin_labels(edges: np.ndarray, feature: pd.Series | None = None) -> list[str]:
    """
    One interval label per bin, written the way the condition reads:

        x < 0.12          the lowest bin
        0.12 < x < 0.34   an interior bin
        0.34 < x          the highest bin

    A single bin -- the baseline node's constant feature -- is labelled 'all'.
    """
    if len(edges) == 0:
        return ['all']
    out = [f'x < {edges[0]:.4g}']
    for a, b in zip(edges[:-1], edges[1:]):
        out.append(f'{a:.4g} < x < {b:.4g}')
    out.append(f'{edges[-1]:.4g} < x')
    return out


def ohlcv_tensor(data: pd.DataFrame) -> torch.Tensor:
    """Convert a stored history to one float64 path in canonical OHLCV order."""
    close = data["close"].to_numpy(float)
    open_ = data["open"].to_numpy(float) if "open" in data else close
    high = data["high"].to_numpy(float) if "high" in data else np.maximum(open_, close)
    low = data["low"].to_numpy(float) if "low" in data else np.minimum(open_, close)
    volume = data["volume"].to_numpy(float) if "volume" in data else np.ones(len(data))
    return tensor_runtime.tensor(np.stack((open_, high, low, close, volume), axis=-1)[None])


def iter_extremes(paths: torch.Tensor, horizons):
    """Shared incremental excursion ladder, yielding one horizon at a time."""
    close, low, high = paths[:, :, 3], paths[:, :, 2], paths[:, :, 1]
    length = close.shape[1]
    reached = 0
    run_lo, run_hi = torch.full_like(close, float("inf")), torch.full_like(close, float("-inf"))
    for horizon in horizons:
        if horizon < reached:
            run_lo.fill_(float("inf"))
            run_hi.fill_(float("-inf"))
            reached = 0
        for offset in range(reached + 1, min(int(horizon), length - 1) + 1):
            n = length - offset
            run_lo[:, :n] = torch.minimum(run_lo[:, :n], low[:, offset:])
            run_hi[:, :n] = torch.maximum(run_hi[:, :n], high[:, offset:])
        reached = min(int(horizon), length - 1)
        n = length - int(horizon)
        lo, hi = torch.full_like(close, float("nan")), torch.full_like(close, float("nan"))
        if n > 0:
            lo[:, :n] = run_lo[:, :n] / close[:, :n] - 1.0
            hi[:, :n] = run_hi[:, :n] / close[:, :n] - 1.0
        yield lo, hi


def forward_extremes_upto(data: pd.DataFrame, t_max: int) -> tuple[np.ndarray, np.ndarray]:
    """Cache the common excursion ladder as (horizon, time) arrays for Stage 1."""
    if t_max < 1:
        raise ValueError("t_max must be positive")
    rows = list(iter_extremes(ohlcv_tensor(data), range(1, t_max + 1)))
    return tuple(torch.cat([row[i] for row in rows], dim=0).cpu().numpy() for i in (0, 1))


def measure_histories(paths, features, deltas, horizons, n_bins, *, edges=None,
                      min_n=MIN_BIN_N, excursions=None, touches=None, baselines=None,
                      bin_indices_cache=None):
    """Return bin edges and an iterator of measured float64 horizon slices.

    Paths have (history, time, OHLCV) axes. Features may have one row, shared
    across histories, or one row per history. Without fixed edges, each history
    receives its own quantiles. Probability slices have (history, barrier, bin,
    1) axes. Each slice includes its own unconditional baseline, measured over
    all eligible market dates independently of feature warm-up.
    Stage 1 may supply its cached excursion ladder for a single history.
    """
    paths = paths.to(dtype=torch.float64) if isinstance(paths, torch.Tensor) else tensor_runtime.tensor(paths)
    x = features.to(dtype=torch.float64) if isinstance(features, torch.Tensor) else tensor_runtime.tensor(features)
    if paths.ndim != 3 or paths.shape[-1] != 5:
        raise ValueError("paths must have (history, time, OHLCV) axes")
    if x.ndim != 2 or x.shape[1] != paths.shape[1] or x.shape[0] not in (1, paths.shape[0]):
        raise ValueError("features must match the path and time axes")
    x = x.expand(paths.shape[:2])
    deltas, horizons = np.asarray(deltas, dtype=float), np.asarray(horizons, dtype=int)
    if deltas.ndim != 1 or horizons.ndim != 1 or np.any(horizons < 1):
        raise ValueError("barriers and horizons must be vectors, with positive horizons")
    if edges is None:
        edges_t = batched_bin_edges(x, n_bins)
    else:
        edges_t = edges.to(dtype=torch.float64) if isinstance(edges, torch.Tensor) else tensor_runtime.tensor(edges)
        edges_t = edges_t.expand(paths.shape[0], -1).contiguous()
    n_bins = edges_t.shape[1] + 1
    indices = (bin_indices(x, edges_t) if bin_indices_cache is None else
               torch.as_tensor(bin_indices_cache, dtype=torch.long, device=paths.device))
    if indices.ndim == 1 and paths.shape[0] == 1:
        indices = indices.unsqueeze(0)
    if indices.shape != paths.shape[:2]:
        raise ValueError("cached bin indices do not match path and time axes")
    if excursions is not None:
        expected = (int(horizons.max()) if len(horizons) else 0, paths.shape[1])
        if paths.shape[0] != 1 or any(value.shape != expected for value in excursions):
            raise ValueError("cached excursions do not match the requested data and horizon grid")
        mins, maxs = (tensor_runtime.tensor(value) for value in excursions)
        extremes = ((mins[t - 1][None], maxs[t - 1][None]) for t in horizons)
    else:
        extremes = iter_extremes(paths, horizons)
    touches_t = None if touches is None else torch.as_tensor(touches, dtype=torch.bool, device=paths.device)
    if touches_t is not None:
        if touches_t.ndim == 3 and paths.shape[0] == 1:
            touches_t = touches_t.unsqueeze(1)
        expected = (len(horizons), paths.shape[0], paths.shape[1], len(deltas))
        if touches_t.shape != expected:
            raise ValueError(f"cached touches do not match measurement axes: {touches_t.shape} vs {expected}")
    baselines_t = None if baselines is None else torch.as_tensor(
        baselines, dtype=paths.dtype, device=paths.device,
    )
    if baselines_t is not None:
        if baselines_t.ndim == 2 and paths.shape[0] == 1:
            baselines_t = baselines_t.unsqueeze(0)
        expected = (paths.shape[0], len(deltas), len(horizons))
        if baselines_t.shape != expected:
            raise ValueError(f"cached baseline does not match measurement axes: {baselines_t.shape} vs {expected}")

    def measurements():
        for horizon_index, (lo, hi) in enumerate(extremes):
            if touches_t is None:
                shared_touch, price_ok = barrier_touch_matrix(lo, hi, deltas)
            else:
                shared_touch = touches_t[horizon_index]
                price_ok = torch.isfinite(lo) & torch.isfinite(hi)
            probabilities, hits, counts, observed = reduce_touch_matrix(
                shared_touch, price_ok, x, indices, n_bins, min_n,
            )
            baseline = (reduce_baseline_touch_matrix(shared_touch, price_ok, paths.dtype, min_n)[0]
                        if baselines_t is None else baselines_t[:, :, horizon_index])
            yield {
                "prob": probabilities.unsqueeze(-1),
                "base": baseline.unsqueeze(-1),
                "hits": hits.unsqueeze(-1),
                "bin_n": counts.unsqueeze(-1), "n_obs": observed.unsqueeze(-1),
            }
    return edges_t, measurements()


def touch_tensor(data: pd.DataFrame, feature: pd.Series, horizons: np.ndarray,
                 Δs: np.ndarray, edges: np.ndarray, min_n: int = MIN_BIN_N,
                 excursions: tuple[np.ndarray, np.ndarray] | None = None,
                 touches=None, baseline=None) -> dict:
    """Stage 1 adapter: collect the shared history measurements into one cube."""
    _, measurements = measure_histories(
        ohlcv_tensor(data), feature.to_numpy(float)[None], Δs, horizons, len(edges) + 1,
        edges=edges, min_n=min_n, excursions=excursions,
        touches=touches, baselines=baseline,
    )
    rows = list(measurements)
    shape = (len(Δs), len(edges) + 1, len(horizons))
    result = {}
    for key, empty_shape in (("prob", shape), ("hits", shape),
                             ("bin_n", shape[1:]), ("n_obs", shape[-1:])):
        values = torch.cat([row[key] for row in rows], dim=-1)[0].cpu().numpy() if rows else np.empty(empty_shape)
        result[key] = values if key == "prob" else values.astype(np.int32)
    values = feature.to_numpy(float)
    return {**result, "Δs": np.asarray(Δs, dtype=float),
            "horizons": np.asarray(horizons, dtype=int), "edges": np.asarray(edges, dtype=float),
            "bin_indices": np.searchsorted(edges, values, side="left").astype(np.uint8)}
