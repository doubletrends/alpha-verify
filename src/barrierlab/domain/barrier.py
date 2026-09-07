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


def cuda_enabled() -> bool:
    return tensor_runtime.device().type == "cuda"


def bin_edges(feature: pd.Series, n_bins: int = 10) -> np.ndarray:
    """
    Interior quantile edges of the feature, so each bin holds ~1/n_bins of the sample.

    Returns at most n_bins-1 edges. A feature with heavy ties -- an integer calendar
    feature, or the constant used by the baseline node -- yields duplicate quantiles,
    so the caller may get fewer bins than asked for. That is correct, not an error.

    An edge only earns its place if it actually splits the sample. Assignment is
    searchsorted-left, so an edge e sends values < e down and values >= e up; it
    therefore needs at least one value on each side, which means min(v) < e <= max(v).
    Without that check a constant feature produces one edge and an empty second bin --
    the baseline node would arrive with a phantom column.
    """
    v = feature.dropna()
    if v.empty:
        return np.array([])
    qs = np.linspace(0, 1, n_bins + 1)[1:-1]
    values = tensor_runtime.tensor(v.to_numpy(float))
    edges = torch.unique(torch.quantile(values, tensor_runtime.tensor(qs))).sort().values
    lo, hi = values.min(), values.max()
    return edges[(edges > lo) & (edges <= hi)].cpu().numpy()


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

    close = data['close'].to_numpy(float)
    low   = (data['low'] if 'low' in data else data['close']).to_numpy(float)
    high  = (data['high'] if 'high' in data else data['close']).to_numpy(float)
    n = len(close)

    run_lo = np.full(n, np.inf)
    run_hi = np.full(n, -np.inf)
    mins = np.full((t_max, n), np.nan)
    maxs = np.full((t_max, n), np.nan)

    for t in range(1, t_max + 1):
        if n - t <= 0:
            break
        run_lo[:n - t] = np.minimum(run_lo[:n - t], low[t:])
        run_hi[:n - t] = np.maximum(run_hi[:n - t], high[t:])
        mins[t - 1, :n - t] = run_lo[:n - t] / close[:n - t] - 1.0
        maxs[t - 1, :n - t] = run_hi[:n - t] / close[:n - t] - 1.0
    return mins, maxs


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

    t_max  = int(np.max(horizons))
    if excursions is None:
        mins, maxs = forward_extremes_upto(data, t_max)
    else:
        mins, maxs = excursions
        expected = (t_max, len(data))
        if mins.shape != expected or maxs.shape != expected:
            raise ValueError("cached excursions do not match the requested data and horizon grid")

    x_all  = feature.to_numpy(float)
    n_bins = len(edges) + 1
    n_th   = len(Δs)
    n_t    = len(horizons)

    prob = np.full((n_th, n_bins, n_t), np.nan)
    hits = np.zeros((n_th, n_bins, n_t), dtype=np.int32)
    bin_n = np.zeros((n_bins, n_t), dtype=np.int32)
    n_obs = np.zeros(n_t, dtype=np.int32)

    for j, t in enumerate(horizons):
        lo_t, hi_t = mins[t - 1], maxs[t - 1]
        ok = ~np.isnan(lo_t) & ~np.isnan(hi_t) & ~np.isnan(x_all)
        if not ok.any():
            continue
        xs, ls, hs = x_all[ok], lo_t[ok], hi_t[ok]
        idx = np.searchsorted(edges, xs) if len(edges) else np.zeros(len(xs), dtype=int)
        counts = np.bincount(idx, minlength=n_bins)
        enough = counts >= min_n
        bin_n[:, j] = counts
        n_obs[j] = len(xs)

        for i, th in enumerate(Δs):
            touched = ((ls <= th) if th < 0 else (hs >= th)).astype(float)
            c = np.bincount(idx, weights=touched, minlength=n_bins)
            hits[i, :, j] = c.astype(np.int32)
            with np.errstate(invalid='ignore', divide='ignore'):
                rates = np.where(enough, c / np.where(counts == 0, 1, counts), np.nan)
            prob[i, :, j] = rates

    return {'prob': prob, 'hits': hits,
            'bin_n': bin_n, 'n_obs': n_obs,
            'Δs': np.asarray(Δs, dtype=float),
            'horizons': np.asarray(horizons, dtype=int),
            'edges': np.asarray(edges, dtype=float)}


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
        ok = torch.isfinite(lo_t) & torch.isfinite(hi_t) & torch.isfinite(x_all)
        if not bool(ok.any()):
            continue
        xs, ls, hs = x_all[ok], lo_t[ok], hi_t[ok]
        idx = torch.bucketize(xs, edges_t, right=False) if len(edges) else torch.zeros(
            len(xs), dtype=torch.int64, device=device
        )
        counts = torch.bincount(idx, minlength=n_bins)
        enough = counts >= min_n
        bin_n[:, j] = counts.to(torch.int32)
        n_obs[j] = len(xs)
        for i, delta in enumerate(deltas_t):
            touched = torch.where(delta < 0, ls <= delta, hs >= delta).to(torch.float64)
            count = torch.bincount(idx, weights=touched, minlength=n_bins)
            hits[i, :, j] = count.to(torch.int32)
            rates = count / torch.clamp(counts, min=1)
            prob[i, :, j] = torch.where(enough, rates, torch.full_like(rates, float("nan")))

    return {
        "prob": prob.cpu().numpy(),
        "hits": hits.cpu().numpy(),
        "bin_n": bin_n.cpu().numpy(),
        "n_obs": n_obs.cpu().numpy(),
        "Δs": np.asarray(Δs, dtype=float),
        "horizons": np.asarray(horizons, dtype=int),
        "edges": np.asarray(edges, dtype=float),
    }


def touch_band(surface: np.ndarray, Δs: np.ndarray, q: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Invert a (n_Δ, n_h) probability surface into the barrier levels touched with
    probability q: the up level and the down level, per horizon.

    The cube answers "how likely is Δ?"; a reader almost always wants the inverse, "how
    far does price get?". Both arms are read off the same surface -- up levels from the
    positive Δ rows, down levels from the negative -- so the answer is one number per
    side per horizon: *at q = 0.5, price touches +4.6% and −3.7% within seven days.*

    P(touch Δ) falls monotonically in |Δ|, so each arm is a decreasing curve and the
    crossing of q is found by linear interpolation between the two bracketing rows.

    This interpolation is for rendering only: it runs on the full cube and never feeds
    a judged measurement. The interpolation is between two measured rows one percentage
    point apart.

    Returns (up, down) as positive magnitudes in Δ units, and NaN in the two places the
    ladder cannot answer: where even the nearest rung is reached less often than q, so
    the crossing lies between 0 and one step and is finer than the grid resolves, and
    where the far end is still above q, so it lies beyond the widest barrier measured.
    Returning 0 for the first case would assert a level the cube never measured.
    """
    th = np.asarray(Δs, dtype=float)
    n_t = surface.shape[1]
    out = []
    for sign in (+1, -1):
        rows = np.flatnonzero(th * sign > 0)
        mag = np.abs(th[rows])
        order = np.argsort(mag)
        rows, mag = rows[order], mag[order]

        levels = np.full(n_t, np.nan)
        for j in range(n_t):
            p = surface[rows, j]
            ok = np.isfinite(p)
            if ok.sum() < 2:
                continue
            m, pv = mag[ok], p[ok]
            if pv[0] < q:            # crossing is finer than one step of the ladder
                levels[j] = np.nan
            elif pv[-1] >= q:        # crossing is beyond the widest barrier measured
                levels[j] = np.nan
            else:
                k = int(np.argmax(pv < q))
                p0, p1 = pv[k - 1], pv[k]
                levels[j] = m[k - 1] + (p0 - q) * (m[k] - m[k - 1]) / (p0 - p1)
        out.append(levels)
    return out[0], out[1]


# ── economic evaluation ───────────────────────────────────────────────────────

def evaluate(
    cube:      dict,
    baseline:  np.ndarray,
    min_dev:   float = 10.0,
    min_bin_n: int = 50,
    min_run:   int = 2,
) -> dict:
    """
    Does this node show a *usable* edge, before asking whether it is a real one?

    Scans the whole cube against the baseline node. Three conditions on the same (Δ, bin, t) cell:

      magnitude    |P(touch | bin) - P(touch)| reaches min_dev percentage points
      support      the bin holds at least min_bin_n observations
      consistency  at least min_run adjacent Δ rows in that (bin, t) column clear
                   min_dev with the same sign

    The consistency requirement is what separates this from a threshold filter. The
    cube has 41 x 10 x 30 cells; the largest single cell under pure noise is well into
    the tens of pp, and picking it is the bin-searching mistake the null test exists to
    catch. A *band* of adjacent barrier levels all deviating the same way is far harder
    to produce by accident, and it is also the only shape that is usable -- an edge at
    exactly -7% and nowhere else is not a stop level, it is an artifact.

    Returns the strongest qualifying cell overall and the best per horizon.
    """
    # `baseline` is the (n_Δ, n_horizons) probability surface of the baseline node
    # -- the unconditional rate, measured as an ordinary node. The comparison is made
    # here and never persisted, so nothing downstream carries two versions of the same
    # measurement.
    #
    # One caveat this inherits: the baseline is measured over every bar, while a node
    # with a long warmup (a 200-bar moving average, say) is measured over fewer. The
    # dropped bars are the oldest ones, so for a long-warmup feature the comparison is
    # very slightly against a different history.
    dev = (cube['prob'] - baseline[:, None, :]) * 100.0
    bin_n = cube['bin_n']
    Δs, horizons = cube['Δs'], cube['horizons']
    n_th, n_bins, n_t = dev.shape

    per_t, best_overall = {}, None
    for j in range(n_t):
        t = int(horizons[j])
        best = None
        for b in range(n_bins):
            if bin_n[b, j] < min_bin_n:
                continue
            col = dev[:, b, j]
            run, start = 0, None
            for i in range(n_th):
                v = col[i]
                if np.isnan(v) or abs(v) < min_dev:
                    run, start = 0, None
                    continue
                if start is not None and np.sign(v) != np.sign(col[start]):
                    run, start = 1, i
                else:
                    if start is None:
                        start = i
                    run += 1
                if run >= min_run:
                    k = int(start + np.argmax(np.abs(col[start:i + 1])))
                    cand = {'horizon': t, 'bin': b, 'Δ': float(Δs[k]),
                            'dev': float(col[k]), 'run': run,
                            'prob': float(cube['prob'][k, b, j]),
                            'base': float(baseline[k, j]),
                            'bin_n': int(bin_n[b, j]), 'hits': int(cube['hits'][k, b, j])}
                    if best is None or abs(cand['dev']) > abs(best['dev']):
                        best = cand
        per_t[t] = best
        if best and (best_overall is None or abs(best['dev']) > abs(best_overall['dev'])):
            best_overall = best

    return {'passed': best_overall is not None, 'best': best_overall,
            'per_horizon': per_t,
            'criteria': {'min_dev': min_dev, 'min_bin_n': min_bin_n, 'min_run': min_run}}
