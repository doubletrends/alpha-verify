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


def bin_labels(edges: np.ndarray, feature: pd.Series) -> list[str]:
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


def forward_extremes_upto(data: pd.DataFrame, t_max: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Worst and best excursion over (t0, t0+t] for every t up to t_max, as (t_max, n) arrays.

    Built incrementally: the window for t is the window for t-1 extended by one bar, so
    the whole ladder costs one pass per horizon instead of one pass per horizon per bar.
    Computing 30 horizons the naive way re-reads the same 465 shifted columns; this
    reads 30.

    Row i holds horizon i+1. Entries whose forward window runs off the end of the
    series are NaN.
    """
    return _forward_extremes_cuda(data, t_max)


def _forward_extremes_cuda(data: pd.DataFrame, t_max: int) -> tuple[np.ndarray, np.ndarray]:
    """CUDA implementation of the shared forward high/low excursion ladder."""
    import torch

    device = tensor_runtime.device()
    close = torch.as_tensor(
        np.array(data["close"], dtype=float, copy=True), dtype=torch.float64, device=device
    )
    low = torch.as_tensor(
        np.array(data["low"] if "low" in data else data["close"], dtype=float, copy=True),
        dtype=torch.float64,
        device=device,
    )
    high = torch.as_tensor(
        np.array(data["high"] if "high" in data else data["close"], dtype=float, copy=True),
        dtype=torch.float64,
        device=device,
    )
    n = len(close)
    run_lo = torch.full((n,), float("inf"), dtype=torch.float64, device=device)
    run_hi = torch.full((n,), float("-inf"), dtype=torch.float64, device=device)
    mins = torch.full((t_max, n), float("nan"), dtype=torch.float64, device=device)
    maxs = torch.full((t_max, n), float("nan"), dtype=torch.float64, device=device)

    for t in range(1, t_max + 1):
        if n - t <= 0:
            break
        run_lo[:n - t] = torch.minimum(run_lo[:n - t], low[t:])
        run_hi[:n - t] = torch.maximum(run_hi[:n - t], high[t:])
        mins[t - 1, :n - t] = run_lo[:n - t] / close[:n - t] - 1.0
        maxs[t - 1, :n - t] = run_hi[:n - t] / close[:n - t] - 1.0
    return mins.cpu().numpy(), maxs.cpu().numpy()


def touch_tensor(
    data:     pd.DataFrame,
    feature:  pd.Series,
    horizons: np.ndarray,
    Δs:   np.ndarray,
    edges:    np.ndarray,
    min_n:    int = MIN_BIN_N,
    excursions: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict:
    """
    The probability cube: P(touch Δ within t | X in bin).

    Shape (n_Δ, n_bins, n_horizons). This is the pipeline's primary measurement and
    the value the workbook shows -- the raw conditional probability, not a deviation
    from anything, so a cell can be read on its own terms ("a -5% touch within 7 days
    happens 50% of the time under this condition") without carrying a base rate in your
    head.

    Nothing here is measured against anything. The unconditional rate is its own node
    -- the `baseline` node, whose feature is constant so every bar falls in one bin --
    and is produced by this same function with no special case.

    Cells whose bin holds fewer than min_n observations are NaN in `prob`; `hits` keeps
    the raw counts so a thin bin stays visible rather than absent.
    """
    return _touch_tensor_cuda(data, feature, horizons, Δs, edges, min_n, excursions)


def _touch_tensor_cuda(
    data: pd.DataFrame,
    feature: pd.Series,
    horizons: np.ndarray,
    Δs: np.ndarray,
    edges: np.ndarray,
    min_n: int,
    excursions: tuple[np.ndarray, np.ndarray] | None,
) -> dict:
    """CUDA touch-tensor kernel with the same output contract as ``touch_tensor``."""
    import torch

    t_max = int(np.max(horizons))
    if excursions is None:
        mins, maxs = _forward_extremes_cuda(data, t_max)
    else:
        mins, maxs = excursions
        expected = (t_max, len(data))
        if mins.shape != expected or maxs.shape != expected:
            raise ValueError("cached excursions do not match the requested data and horizon grid")

    device = tensor_runtime.device()
    mins_t = torch.as_tensor(np.array(mins, copy=True), dtype=torch.float64, device=device)
    maxs_t = torch.as_tensor(np.array(maxs, copy=True), dtype=torch.float64, device=device)
    x_all = torch.as_tensor(
        np.array(feature, dtype=float, copy=True), dtype=torch.float64, device=device
    )
    edges_t = torch.as_tensor(edges, dtype=torch.float64, device=device)
    deltas_t = torch.as_tensor(Δs, dtype=torch.float64, device=device)
    n_bins, n_th, n_t = len(edges) + 1, len(Δs), len(horizons)

    prob = torch.full((n_th, n_bins, n_t), float("nan"), dtype=torch.float64, device=device)
    hits = torch.zeros((n_th, n_bins, n_t), dtype=torch.int32, device=device)
    bin_n = torch.zeros((n_bins, n_t), dtype=torch.int32, device=device)
    n_obs = torch.zeros(n_t, dtype=torch.int32, device=device)

    for j, t in enumerate(horizons):
        lo_t, hi_t = mins_t[t - 1], maxs_t[t - 1]
        idx = bin_indices(x_all, edges_t)
        for i, delta in enumerate(deltas_t):
            rates, count, counts, observed = touch_rates(
                lo_t[None, :], hi_t[None, :], x_all[None, :], idx[None, :],
                n_bins, delta, min_n,
            )
            bin_n[:, j] = counts[0].to(torch.int32)
            n_obs[j] = observed[0]
            hits[i, :, j] = count[0].to(torch.int32)
            prob[i, :, j] = rates[0]

    return {
        "prob": prob.cpu().numpy(),
        "hits": hits.cpu().numpy(),
        "bin_n": bin_n.cpu().numpy(),
        "n_obs": n_obs.cpu().numpy(),
        "Δs": np.asarray(Δs, dtype=float),
        "horizons": np.asarray(horizons, dtype=int),
        "edges": np.asarray(edges, dtype=float),
    }
