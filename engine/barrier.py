"""
Barrier-touch surfaces: will price reach ±theta within h bars, given a condition?

This is the engine's single measurement. Everything the pipeline used to compute as a
separate "outcome" is a row of it: the -10% drawdown surface is the theta = -10% row,
the +10% runup surface is the +10% row, and the asymmetry between them is the two rows
read against each other in the same column. There is no outcome registry any more --
theta *is* the axis.

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

# Below this many observations a bin's rate is not worth reporting.
MIN_BIN_N = 30


def theta_levels(lo: float = -0.20, hi: float = 0.20, step: float = 0.01) -> np.ndarray:
    """
    Barrier levels for the rows of a surface, excluding 0.

    theta = 0 is degenerate -- price touches its own starting level essentially always
    -- so it carries no information and is dropped rather than printed as a row of 100%.
    """
    n = int(round((hi - lo) / step)) + 1
    levels = np.round(np.linspace(lo, hi, n), 10)
    return levels[np.abs(levels) > step / 2]


def bin_edges(feature: pd.Series, n_bins: int = 10) -> np.ndarray:
    """
    Interior quantile edges of the feature, so each bin holds ~1/n_bins of the sample.

    Returns n_bins-1 edges. Duplicate edges (a feature with mass at a single value,
    e.g. an integer calendar feature) are collapsed, so the caller may get fewer bins
    than asked for -- which is correct, not an error.
    """
    v = feature.dropna()
    if v.empty:
        return np.array([])
    qs = np.linspace(0, 1, n_bins + 1)[1:-1]
    return np.unique(np.quantile(v, qs))


def bin_labels(edges: np.ndarray, feature: pd.Series) -> list[str]:
    """Human-readable range label per bin, e.g. 'x < 0.12' / '0.12..0.34' / 'x > 0.34'."""
    if len(edges) == 0:
        return ['all']
    out = [f'x < {edges[0]:.4g}']
    for a, b in zip(edges[:-1], edges[1:]):
        out.append(f'{a:.4g}..{b:.4g}')
    out.append(f'x > {edges[-1]:.4g}')
    return out


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


def touch_matrix(
    data:     pd.DataFrame,
    feature:  pd.Series,
    h:        int,
    thetas:   np.ndarray,
    edges:    np.ndarray,
    min_n:    int = MIN_BIN_N,
) -> dict:
    """
    P(touch theta within h | feature in bin), for every theta and every bin.

    Returns a dict of aligned arrays:
        prob     (n_theta, n_bins)  conditional touch probability, 0..1
        hits     (n_theta, n_bins)  event counts behind each probability
        base     (n_theta,)         unconditional touch probability
        dev      (n_theta, n_bins)  prob - base, in percentage points
        bin_n    (n_bins,)          observations per bin
        thetas, n_obs

    A negative theta asks whether the *low* fell to it; a positive theta whether the
    *high* rose to it. Cells backed by fewer than min_n observations are NaN in `prob`
    and `dev` -- the counts stay in `hits` so a thin column is visible rather than
    silently absent.
    """
    fwd_low, fwd_high = forward_extremes(data, h)
    df = pd.concat([feature.rename('x'), fwd_low.rename('lo'), fwd_high.rename('hi')],
                   axis=1).dropna()

    n_bins = len(edges) + 1
    n_th   = len(thetas)
    if df.empty:
        nan = np.full((n_th, n_bins), np.nan)
        return {'prob': nan, 'hits': np.zeros((n_th, n_bins), int), 'dev': nan.copy(),
                'base': np.full(n_th, np.nan), 'bin_n': np.zeros(n_bins, int),
                'thetas': thetas, 'n_obs': 0}

    x   = df['x'].to_numpy(float)
    lo  = df['lo'].to_numpy(float)
    hi  = df['hi'].to_numpy(float)
    idx = np.searchsorted(edges, x) if len(edges) else np.zeros(len(x), dtype=int)

    bin_n = np.bincount(idx, minlength=n_bins)
    ok    = bin_n >= min_n

    prob = np.full((n_th, n_bins), np.nan)
    hits = np.zeros((n_th, n_bins), dtype=int)
    base = np.empty(n_th)

    for i, th in enumerate(thetas):
        touched = (lo <= th) if th < 0 else (hi >= th)
        t = touched.astype(float)
        base[i] = t.mean()
        counts  = np.bincount(idx, weights=t, minlength=n_bins)
        hits[i] = counts.astype(int)
        with np.errstate(invalid='ignore', divide='ignore'):
            rates = np.where(ok, counts / np.where(bin_n == 0, 1, bin_n), np.nan)
        prob[i] = rates

    dev = (prob - base[:, None]) * 100.0
    return {'prob': prob, 'hits': hits, 'dev': dev, 'base': base,
            'bin_n': bin_n, 'thetas': thetas, 'n_obs': int(len(df))}


# ── economic evaluation ───────────────────────────────────────────────────────

def evaluate(
    surfaces:      dict,
    min_dev:       float = 10.0,
    min_bin_n:     int = 50,
    min_run:       int = 2,
) -> dict:
    """
    Does this node show a *usable* edge, before asking whether it is a real one?

    Three conditions, all on the same (theta, bin) cell:

      magnitude    |P(touch | bin) - P(touch)| reaches min_dev percentage points
      support      the bin holds at least min_bin_n observations
      consistency  at least min_run adjacent theta rows in that column clear min_dev
                   with the same sign

    The consistency requirement is what separates this from a threshold filter. A
    surface has ~40 x 10 cells per horizon; the largest single cell in pure noise is
    comfortably into the teens of pp, and picking it is the same bin-searching mistake
    the null test exists to catch. A *band* of adjacent barrier levels all deviating
    the same way is much harder to produce by accident, and it is also the only shape
    that is tradeable -- an edge at exactly -7% and nowhere else is not a stop level,
    it is an artifact.

    Returns the best qualifying cell and a pass flag, per horizon and overall.
    """
    per_h, best_overall = {}, None
    for label, s in surfaces.items():
        dev, bin_n = s['dev'], s['bin_n']
        n_th, n_bins = dev.shape
        best = None
        for b in range(n_bins):
            if bin_n[b] < min_bin_n:
                continue
            col = dev[:, b]
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
                    j = int(start + np.argmax(np.abs(col[start:i + 1])))
                    cand = {'horizon': label, 'bin': b, 'theta': float(s['thetas'][j]),
                            'dev': float(col[j]), 'run': run,
                            'bin_n': int(bin_n[b]), 'hits': int(s['hits'][j, b])}
                    if best is None or abs(cand['dev']) > abs(best['dev']):
                        best = cand
        per_h[label] = best
        if best and (best_overall is None or abs(best['dev']) > abs(best_overall['dev'])):
            best_overall = best

    return {'passed': best_overall is not None, 'best': best_overall, 'per_horizon': per_h,
            'criteria': {'min_dev': min_dev, 'min_bin_n': min_bin_n, 'min_run': min_run}}


# ── statistical validation ────────────────────────────────────────────────────

def _peak_dev(touched: np.ndarray, onehot: np.ndarray, bin_n: np.ndarray,
              base: np.ndarray, min_n: int) -> float:
    """
    Max |P(touch | bin) - P(touch)| over every (theta, bin) cell, in pp.

    `touched` is (n_theta, n_obs) and does not depend on the feature, so the whole
    surface is one matrix product against the bin one-hot rather than a loop over
    thetas -- which is what makes tens of thousands of shuffles affordable.
    """
    counts = touched @ onehot                      # (n_theta, n_bins)
    ok     = bin_n >= min_n
    if not ok.any():
        return 0.0
    rates = counts[:, ok] / bin_n[ok]
    return float(np.abs(rates - base[:, None]).max() * 100.0)


def null_test(
    data:     pd.DataFrame,
    feature:  pd.Series,
    h:        int,
    thetas:   np.ndarray,
    n_bins:   int = 10,
    min_n:    int = MIN_BIN_N,
    n_shifts: int = 2000,
    seed:     int = 0,
) -> dict:
    """
    Is the strongest cell of a node's surface more than the search that found it?

    The headline of a surface is a maximum over ~40 barrier levels x ~10 bins. That is
    400 noisy estimates, and the largest of them is well into the tens of pp even when
    the feature carries nothing at all -- a far bigger bias than the single-threshold
    scans this replaces, because the search is two-dimensional.

    The null circularly shifts the *feature* against price. Shifting the feature rather
    than the outcome keeps the two forward-excursion series consistent with each other
    and with the price path, preserves the feature's own autocorrelation, and destroys
    only its alignment with the future -- which is exactly the thing being tested.
    """
    fwd_low, fwd_high = forward_extremes(data, h)
    df = pd.concat([feature.rename('x'), fwd_low.rename('lo'), fwd_high.rename('hi')],
                   axis=1).dropna()
    guard = max(3 * h, 200)
    if len(df) < 2 * guard + 1:
        return {'real': np.nan, 'null_median': np.nan, 'null_p95': np.nan,
                'p_value': np.nan, 'n_obs': len(df), 'n_shifts': 0}

    x  = df['x'].to_numpy(float)
    lo = df['lo'].to_numpy(float)
    hi = df['hi'].to_numpy(float)
    n  = len(x)

    # touched[i, t] — whether barrier thetas[i] was reached from bar t. Independent of
    # the feature, so it is built once and reused for every shuffle.
    touched = np.empty((len(thetas), n), dtype=np.float32)
    for i, th in enumerate(thetas):
        touched[i] = (lo <= th) if th < 0 else (hi >= th)
    base = touched.mean(axis=1).astype(np.float64)

    # A circular shift permutes the feature, so its quantiles -- and therefore the bin
    # edges -- are identical under every shuffle. Computing them once is both correct
    # and the difference between a fast sweep and an unusable one.
    edges = bin_edges(pd.Series(x), n_bins)
    nb    = len(edges) + 1
    rows  = np.arange(n)

    def stat(xv: np.ndarray) -> float:
        idx = np.searchsorted(edges, xv) if len(edges) else np.zeros(n, dtype=int)
        onehot = np.zeros((n, nb), dtype=np.float32)
        onehot[rows, idx] = 1.0
        return _peak_dev(touched, onehot, onehot.sum(axis=0).astype(np.float64), base, min_n)

    real = stat(x)
    rng    = np.random.default_rng(seed)
    shifts = rng.integers(guard, n - guard, size=n_shifts)
    null   = np.array([stat(np.roll(x, int(s))) for s in shifts])

    n_ge = int((null >= real).sum())
    return {
        'real':        round(real, 3),
        'null_median': round(float(np.median(null)), 3),
        'null_p95':    round(float(np.percentile(null, 95)), 3),
        'p_value':     round((1.0 + n_ge) / (1.0 + n_shifts), 6),
        'p_floor':     round(1.0 / (1.0 + n_shifts), 6),
        'n_obs':       int(n),
        'n_shifts':    int(n_shifts),
    }


def verdict(p_value: float, alpha: float = 0.05, n_tests: int = 1,
            p_floor: float = 0.0) -> str:
    """
    structure    clears alpha/n_tests -- survives correction for the size of the sweep
    nominal      clears alpha alone
    underpowered sits at the resolution floor; more shuffles might resolve it
    noise        indistinguishable from the null
    """
    if pd.isna(p_value):
        return 'insufficient'
    corr = alpha / max(n_tests, 1)
    if p_value <= corr:
        return 'structure'
    if p_value <= p_floor and p_floor > corr:
        return 'underpowered'
    if p_value <= alpha:
        return 'nominal'
    return 'noise'
