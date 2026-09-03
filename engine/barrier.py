"""
Barrier-touch surfaces: will price reach ±θ within h bars, given a condition?

This is the engine's single measurement: the probability that price reaches a barrier,
given a condition. Everything the pipeline used to compute as a separate "outcome" is a
row of it -- the -10% drawdown surface is the θ = -10% row, the +10% runup surface
is the +10% row, and the asymmetry between them is the two rows read against each other
in the same column. There is no outcome registry any more; θ *is* the axis.

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

import json
from pathlib import Path

import numpy as np
import pandas as pd

# Below this many observations a bin's rate is not worth reporting.
MIN_BIN_N = 30


def theta_levels(lo: float = -0.20, hi: float = 0.20, step: float = 0.01) -> np.ndarray:
    """
    Barrier levels for the rows of a surface, ascending, including 0.

    θ = 0 is near-degenerate -- it asks whether the high ever returned to the entry
    close, which is almost always true -- but it is kept so the ladder is complete and
    symmetric, and so the row above and below it are read against a real boundary
    rather than a gap.
    """
    n = int(round((hi - lo) / step)) + 1
    return np.round(np.linspace(lo, hi, n), 10)


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
    edges = np.unique(np.quantile(v, qs))
    lo, hi = float(v.min()), float(v.max())
    return edges[(edges > lo) & (edges <= hi)]


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


def forward_extremes_upto(data: pd.DataFrame, h_max: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Worst and best excursion over (t, t+h] for every h up to h_max, as (h_max, n) arrays.

    Built incrementally: the window for h is the window for h-1 extended by one bar, so
    the whole ladder costs one pass per horizon instead of one pass per horizon per bar.
    Computing 30 horizons the naive way re-reads the same 465 shifted columns; this
    reads 30.

    Row i holds horizon i+1. Entries whose forward window runs off the end of the
    series are NaN.
    """
    close = data['close'].to_numpy(float)
    low   = (data['low'] if 'low' in data else data['close']).to_numpy(float)
    high  = (data['high'] if 'high' in data else data['close']).to_numpy(float)
    n = len(close)

    run_lo = np.full(n, np.inf)
    run_hi = np.full(n, -np.inf)
    mins = np.full((h_max, n), np.nan)
    maxs = np.full((h_max, n), np.nan)

    for h in range(1, h_max + 1):
        if n - h <= 0:
            break
        run_lo[:n - h] = np.minimum(run_lo[:n - h], low[h:])
        run_hi[:n - h] = np.maximum(run_hi[:n - h], high[h:])
        mins[h - 1, :n - h] = run_lo[:n - h] / close[:n - h] - 1.0
        maxs[h - 1, :n - h] = run_hi[:n - h] / close[:n - h] - 1.0
    return mins, maxs


def touch_tensor(
    data:     pd.DataFrame,
    feature:  pd.Series,
    horizons: np.ndarray,
    thetas:   np.ndarray,
    edges:    np.ndarray,
    min_n:    int = MIN_BIN_N,
) -> dict:
    """
    The probability cube: P(touch θ within h | X in bin).

    Shape (n_theta, n_bins, n_horizons). This is the pipeline's primary measurement and
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
    h_max  = int(np.max(horizons))
    mins, maxs = forward_extremes_upto(data, h_max)

    x_all  = feature.to_numpy(float)
    n_bins = len(edges) + 1
    n_th   = len(thetas)
    n_h    = len(horizons)

    prob = np.full((n_th, n_bins, n_h), np.nan)
    hits = np.zeros((n_th, n_bins, n_h), dtype=np.int32)
    bin_n = np.zeros((n_bins, n_h), dtype=np.int32)
    n_obs = np.zeros(n_h, dtype=np.int32)

    for j, h in enumerate(horizons):
        lo_h, hi_h = mins[h - 1], maxs[h - 1]
        ok = ~np.isnan(lo_h) & ~np.isnan(hi_h) & ~np.isnan(x_all)
        if not ok.any():
            continue
        xs, ls, hs = x_all[ok], lo_h[ok], hi_h[ok]
        idx = np.searchsorted(edges, xs) if len(edges) else np.zeros(len(xs), dtype=int)
        counts = np.bincount(idx, minlength=n_bins)
        enough = counts >= min_n
        bin_n[:, j] = counts
        n_obs[j] = len(xs)

        for i, th in enumerate(thetas):
            t = ((ls <= th) if th < 0 else (hs >= th)).astype(float)
            c = np.bincount(idx, weights=t, minlength=n_bins)
            hits[i, :, j] = c.astype(np.int32)
            with np.errstate(invalid='ignore', divide='ignore'):
                rates = np.where(enough, c / np.where(counts == 0, 1, counts), np.nan)
            prob[i, :, j] = rates

    return {'prob': prob, 'hits': hits,
            'bin_n': bin_n, 'n_obs': n_obs,
            'thetas': np.asarray(thetas, dtype=float),
            'horizons': np.asarray(horizons, dtype=int),
            'edges': np.asarray(edges, dtype=float)}


def forward_extremes(data: pd.DataFrame, h: int) -> tuple[pd.Series, pd.Series]:
    """
    Worst and best excursion over (t, t+h], as returns relative to close_t.

    Uses low/high, so the result is what a resting stop or limit order would actually
    experience. The window opens at t+1: a barrier cannot be touched on the bar the
    condition is read on. NaN where the full window is not realized.
    """
    close = data['close']
    low   = data['low'] if 'low' in data else close
    high  = data['high'] if 'high' in data else close

    lows  = pd.concat([low.shift(-k) for k in range(1, h + 1)], axis=1).min(axis=1)
    highs = pd.concat([high.shift(-k) for k in range(1, h + 1)], axis=1).max(axis=1)
    valid = close.shift(-h).notna()
    return (lows / close - 1.0).where(valid), (highs / close - 1.0).where(valid)


def summarize(cube: dict, thetas: np.ndarray, horizons: np.ndarray,
              tol: float = 1e-9) -> dict:
    """
    Select a coarse sub-grid of a full cube, without re-measuring anything.

    The full cube is the faithful record: 41 barrier levels a percentage point apart,
    every horizon from 1 to 30. Adjacent rows and adjacent columns of that are very
    nearly the same measurement, which matters when the surface is *judged* rather than
    read -- a peak taken over 12,300 cells carries a large noise ceiling by construction,
    and counting those cells as 12,300 independent tests overstates the size of the
    search in both directions at once.

    The summary keeps the same three axes and the same format, so the renderer and the
    null both work on it unchanged. It is pure index selection: every requested value
    must already exist in the full cube, or this raises rather than interpolating.
    Silently interpolating would make the summary a second measurement, and then the two
    could disagree.
    """
    def pick(want: np.ndarray, have: np.ndarray, name: str) -> np.ndarray:
        idx = []
        for v in want:
            hits = np.flatnonzero(np.abs(have - v) <= tol)
            if not len(hits):
                raise ValueError(
                    f'summary {name} {v!r} is not on the full grid '
                    f'({have.min()}..{have.max()}); it must be a subset, not an interpolation')
            idx.append(int(hits[0]))
        return np.array(idx, dtype=int)

    ti = pick(np.asarray(thetas, float), cube['thetas'], 'theta')
    hi = pick(np.asarray(horizons, float), cube['horizons'].astype(float), 'horizon')

    return {
        'prob':     cube['prob'][np.ix_(ti, range(cube['prob'].shape[1]), hi)],
        'hits':     cube['hits'][np.ix_(ti, range(cube['hits'].shape[1]), hi)],
        'bin_n':    cube['bin_n'][:, hi],
        'n_obs':    cube['n_obs'][hi],
        'thetas':   cube['thetas'][ti],
        'horizons': cube['horizons'][hi],
        'edges':    cube['edges'],
    }


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

    Scans the whole cube against the baseline node. Three conditions on the same (θ, bin, h) cell:

      magnitude    |P(touch | bin) - P(touch)| reaches min_dev percentage points
      support      the bin holds at least min_bin_n observations
      consistency  at least min_run adjacent θ rows in that (bin, h) column clear
                   min_dev with the same sign

    The consistency requirement is what separates this from a threshold filter. The
    cube has 41 x 10 x 30 cells; the largest single cell under pure noise is well into
    the tens of pp, and picking it is the bin-searching mistake the null test exists to
    catch. A *band* of adjacent barrier levels all deviating the same way is far harder
    to produce by accident, and it is also the only shape that is usable -- an edge at
    exactly -7% and nowhere else is not a stop level, it is an artifact.

    Returns the strongest qualifying cell overall and the best per horizon.
    """
    # `baseline` is the (n_theta, n_horizons) probability surface of the baseline node
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
    thetas, horizons = cube['thetas'], cube['horizons']
    n_th, n_bins, n_h = dev.shape

    per_h, best_overall = {}, None
    for j in range(n_h):
        h = int(horizons[j])
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
                    cand = {'horizon': h, 'bin': b, 'theta': float(thetas[k]),
                            'dev': float(col[k]), 'run': run,
                            'prob': float(cube['prob'][k, b, j]),
                            'base': float(baseline[k, j]),
                            'bin_n': int(bin_n[b, j]), 'hits': int(cube['hits'][k, b, j])}
                    if best is None or abs(cand['dev']) > abs(best['dev']):
                        best = cand
        per_h[h] = best
        if best and (best_overall is None or abs(best['dev']) > abs(best_overall['dev'])):
            best_overall = best

    return {'passed': best_overall is not None, 'best': best_overall,
            'per_horizon': per_h,
            'criteria': {'min_dev': min_dev, 'min_bin_n': min_bin_n, 'min_run': min_run}}


def save_cube(cube: dict, path: Path, meta: dict) -> None:
    """
    Write a cube to a compressed .npz.

    Self-describing: the probability array travels with its three coordinate vectors
    and the counts it was built from, so a reader needs nothing but the file. Loads in milliseconds and stays well under a megabyte, which
    matters because there is one per node.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        prob=cube['prob'].astype(np.float32),
        hits=cube['hits'],
        bin_n=cube['bin_n'],
        n_obs=cube['n_obs'],
        thetas=cube['thetas'],
        horizons=cube['horizons'],
        edges=cube['edges'],
        meta=np.array(json.dumps(meta)),
    )


def load_cube(path: Path) -> dict:
    """Read a cube back, with `meta` decoded and the rate arrays widened to float64."""
    z = np.load(path, allow_pickle=False)
    out = {k: z[k] for k in z.files if k != 'meta'}
    for k in ('prob',):
        out[k] = out[k].astype(np.float64)
    out['meta'] = json.loads(str(z['meta']))
    return out
