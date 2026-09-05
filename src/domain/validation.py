"""
Stage three: is a node's surface more than the search that found it?

A cube is 41 barrier levels x 10 bins x 30 horizons. Reading the largest cell off that
and calling it an edge measures how many cells were searched, not whether the feature
carries anything -- the maximum of ~400 noisy rates per horizon lands well into the
tens of percentage points on a feature that knows nothing.

The null shifts the feature circularly against price. Each shift preserves the feature's
own autocorrelation and destroys only its alignment with the future, which is precisely
the thing under test.

Two things about this implementation are worth knowing.

**It is exact, not sampled.** Writing counts[Δ, bin] as a function of shift makes it
a circular cross-correlation between the touch indicator and the bin-membership mask, so
one FFT produces every shift at once -- about 70x faster than resampling, and it returns
the whole permutation distribution rather than a draw from it.

**The floor is real.** There are only n distinct circular shifts, so the finest p-value
obtainable is 1/(n+1) -- roughly 2.4e-4 on eleven years of daily bars. Drawing 20,000
random shifts from a group of 4,214 and reporting p = 1/20001 claims a resolution the
data cannot produce; it inflates significance by about 5x. Because that floor sits above
the Bonferroni threshold for this sweep size, family-wise correction is too harsh here,
and `bh` below controls false discovery rate instead.

The reference the null measures against is the node's *own* sample rate, not the baseline
node's. Shifting leaves the marginal untouched, so the node's own rate is the quantity
that stays fixed under the null and the only one the test can be built on. The economic
filter in `barrier.evaluate` asks a different question -- is this unusual against the
workspace reference -- and correctly uses the baseline node for it.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# One threshold, defined by the module that does the binning, so measurement and
# validation partition the sample identically.
from domain.barrier import MIN_BIN_N

# Shifts smaller than this leave the series almost aligned with itself and are not
# honest null draws; the same applies to shifts near a full wrap.
EDGE_GUARD = 200


def _dev_all_shifts(touched: np.ndarray, idx: np.ndarray, bin_n: np.ndarray,
                    n_bins: int) -> np.ndarray:
    """
    Deviation from the sample rate for every (Δ, bin, shift), in percentage points.

    counts[i, b, s] = sum_t touched[i, t] * 1[bin of (t - s) == b]

    which is a circular cross-correlation in s, so the whole shift axis comes out of one
    inverse FFT. Shift 0 is the identity and therefore holds the real, unshifted surface.
    """
    n = touched.shape[1]
    mask = np.zeros((n_bins, n))
    mask[idx, np.arange(n)] = 1.0

    ft = np.fft.rfft(touched, axis=1)
    fm = np.conj(np.fft.rfft(mask, axis=1))
    counts = np.fft.irfft(ft[:, None, :] * fm[None, :, :], n=n, axis=2)

    # These are counts of 0/1 indicators and so are exact integers; the FFT returns them
    # with ~1e-13 of roundoff. Rounding restores exactness, which matters most in the
    # degenerate case: a single-bin cube has a deviation of identically zero, and
    # without this the null would be comparing one speck of numerical noise against
    # another and returning a meaningless p-value instead of 1.
    counts = np.rint(counts)

    base = touched.mean(axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        rates = counts / np.where(bin_n == 0, np.nan, bin_n)[None, :, None]
    return (rates - base[:, None, None]) * 100.0


def _usable_shifts(n: int, guard: int) -> np.ndarray:
    """Boolean mask over the shift axis, excluding the identity and near-wraps."""
    s = np.arange(n)
    return (s >= guard) & (s <= n - guard)


def null_surface(
    touched:         np.ndarray,
    idx:             np.ndarray,
    n_bins:          int,
    min_n:           int = MIN_BIN_N,
    guard:           int = EDGE_GUARD,
) -> dict:
    """
    Null-test one horizon's surface, per cell and as a whole.

    Returns
        cell_real   (n_Δ, n_bins)  signed deviation from the sample rate, pp
        cell_p      (n_Δ, n_bins)  pointwise p-value of |deviation|
        cell_p95    (n_Δ, n_bins)  95th percentile of the cell's own null
        peak_real   scalar             max |deviation| over the surface
        peak_p      scalar             p-value of that maximum
        peak_p95    scalar             95th percentile of the maximum's null
        n_shifts    int                usable shifts, so the p-value floor is 1/(n+1)
    `cell_p` is pointwise and nothing else. With 12,300 cells, ~615 sit below 0.05 by
    chance, so it is evidence for *reading* a surface and never a discovery criterion.
    `peak_p` is the statistic that accounts for the search, and it is the one the gate
    corrects across.
    """
    n = touched.shape[1]
    bin_n = np.bincount(idx, minlength=n_bins).astype(float)
    enough = bin_n >= min_n

    signed = _dev_all_shifts(touched, idx, bin_n, n_bins)
    signed[:, ~enough, :] = np.nan

    usable = _usable_shifts(n, guard)
    n_use = int(usable.sum())
    if n_use < 50:
        nan2 = np.full(signed.shape[:2], np.nan)
        nan1 = np.full(n_bins, np.nan)
        return {'cell_real': nan2, 'cell_p': nan2.copy(), 'cell_p95': nan2.copy(),
                'sheet_peak_real': nan1, 'sheet_peak_p': nan1.copy(),
                'sheet_peak_p95': nan1.copy(),
                'peak_real': np.nan, 'peak_p': np.nan, 'peak_p95': np.nan, 'n_shifts': 0}

    # the reported deviation keeps its sign -- which way the condition moves the barrier
    # is the whole point -- while the test itself is two-sided and compares magnitudes
    real_signed = signed[:, :, 0]
    dev = np.abs(signed)
    real = dev[:, :, 0]
    null = dev[:, :, usable]

    # add-one (Davison & Hinkley): the observed value is itself one draw from the null,
    # so a permutation p-value is never exactly zero
    ge = (null >= real[:, :, None]).sum(axis=2)
    cell_p = (1.0 + ge) / (1.0 + n_use)
    cell_p[~np.isfinite(real)] = np.nan
    with warnings.catch_warnings():
        # bins below min_n are all-NaN by design; nanpercentile says so loudly
        warnings.simplefilter('ignore', category=RuntimeWarning)
        cell_p95 = np.nanpercentile(null, 95, axis=2)

    with np.errstate(invalid='ignore'):
        peak_by_shift = np.nanmax(dev.reshape(-1, n), axis=0)
    peak_real = float(peak_by_shift[0])
    peak_null = peak_by_shift[usable]
    peak_p = float((1.0 + (peak_null >= peak_real).sum()) / (1.0 + n_use))

    with np.errstate(invalid='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        sheet_peak_by_shift = np.nanmax(dev, axis=0)
    sheet_real = sheet_peak_by_shift[:, 0]
    sheet_null = sheet_peak_by_shift[:, usable]
    sheet_p = np.full(n_bins, np.nan)
    sheet_p95 = np.full(n_bins, np.nan)
    for b in range(n_bins):
        if not np.isfinite(sheet_real[b]):
            continue
        sheet_p[b] = (1.0 + (sheet_null[b] >= sheet_real[b]).sum()) / (1.0 + n_use)
        sheet_p95[b] = np.percentile(sheet_null[b], 95)

    return {
        'cell_real': real_signed,
        'cell_p':    cell_p,
        'cell_p95':  cell_p95,
        'sheet_peak_real': sheet_real,
        'sheet_peak_p': sheet_p,
        'sheet_peak_p95': sheet_p95,
        'peak_real': peak_real,
        'peak_p':    peak_p,
        'peak_p95':  float(np.percentile(peak_null, 95)),
        'n_shifts':  n_use,
    }


def validate_node(
    data:          pd.DataFrame,
    feature:       pd.Series,
    horizons:      np.ndarray,
    Δs:        np.ndarray,
    edges:         np.ndarray,
    min_n:         int = MIN_BIN_N,
    guard:         int = EDGE_GUARD,
) -> dict:
    """
    Null-test every horizon of a node, returning arrays shaped like its cube.

    The bin edges are the node's own, passed in rather than recomputed, so validation
    and the cube partition the sample identically.

    """
    from domain import barrier

    t_max = int(np.max(horizons))
    mins, maxs = barrier.forward_extremes_upto(data, t_max)
    x_all = feature.to_numpy(float)
    n_bins = len(edges) + 1
    n_th, n_t = len(Δs), len(horizons)

    out = {k: np.full((n_th, n_bins, n_t), np.nan)
           for k in ('cell_real', 'cell_p', 'cell_p95')}
    for k in ('peak_real', 'peak_p', 'peak_p95'):
        out[k] = np.full(n_t, np.nan)
    for k in ('sheet_peak_real', 'sheet_peak_p', 'sheet_peak_p95'):
        out[k] = np.full((n_bins, n_t), np.nan)
    out['n_shifts'] = np.zeros(n_t, dtype=np.int32)

    for j, t in enumerate(horizons):
        lo_t, hi_t = mins[t - 1], maxs[t - 1]
        ok = ~np.isnan(lo_t) & ~np.isnan(hi_t) & ~np.isnan(x_all)
        if ok.sum() < 2 * guard + 1:
            continue
        xs, ls, hs = x_all[ok], lo_t[ok], hi_t[ok]
        idx = np.searchsorted(edges, xs) if len(edges) else np.zeros(len(xs), dtype=int)

        touched = np.empty((n_th, len(xs)))
        for i, th in enumerate(Δs):
            touched[i] = (ls <= th) if th < 0 else (hs >= th)

        r = null_surface(touched, idx, n_bins, min_n, guard)
        for k in ('cell_real', 'cell_p', 'cell_p95'):
            out[k][:, :, j] = r[k]
        for k in ('sheet_peak_real', 'sheet_peak_p', 'sheet_peak_p95'):
            out[k][:, j] = r[k]
        for k in ('peak_real', 'peak_p', 'peak_p95'):
            out[k][j] = r[k]
        out['n_shifts'][j] = r['n_shifts']

    out['Δs']   = np.asarray(Δs, dtype=float)
    out['horizons'] = np.asarray(horizons, dtype=int)
    # carried so the renderer works straight off this dict, not only off a reloaded
    # artifact; `save` writes the richer meta passed to it
    out['meta']     = {'bin_labels': barrier.bin_labels(edges, feature)}
    return out


def peak_shift_distribution(
    data:     pd.DataFrame,
    feature:  pd.Series,
    horizon:  int,
    Δs:   np.ndarray,
    edges:    np.ndarray,
    bin_index: int | None = None,
    min_n:    int = MIN_BIN_N,
    guard:    int = EDGE_GUARD,
) -> dict:
    """
    The whole null for one horizon, not just its summary: max |deviation| over the
    surface, or over one selected bin sheet, for every usable circular shift beside the
    observed value.

    `null_surface` reduces this to three numbers because that is all the gate needs. The
    distribution itself is what makes the stage legible -- a histogram of ~3,800 shifts
    with the real surface sitting outside all of them says in one glance what a p-value
    at the resolution floor means, and why that floor exists at all.

    Returns {null, observed, p95, p_value, n_shifts, floor}.
    """
    from domain import barrier

    mins, maxs = barrier.forward_extremes_upto(data, int(horizon))
    lo_t, hi_t = mins[int(horizon) - 1], maxs[int(horizon) - 1]
    x_all = feature.to_numpy(float)
    ok = ~np.isnan(lo_t) & ~np.isnan(hi_t) & ~np.isnan(x_all)
    xs, ls, hs = x_all[ok], lo_t[ok], hi_t[ok]

    n_bins = len(edges) + 1
    idx = np.searchsorted(edges, xs) if len(edges) else np.zeros(len(xs), dtype=int)
    bin_n = np.bincount(idx, minlength=n_bins).astype(float)

    touched = np.empty((len(Δs), len(xs)))
    for i, th in enumerate(Δs):
        touched[i] = (ls <= th) if th < 0 else (hs >= th)

    signed = _dev_all_shifts(touched, idx, bin_n, n_bins)
    signed[:, bin_n < min_n, :] = np.nan
    if bin_index is not None:
        keep = np.zeros(n_bins, dtype=bool)
        keep[int(bin_index)] = True
        signed[:, ~keep, :] = np.nan

    usable = _usable_shifts(len(xs), guard)
    with np.errstate(invalid='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        peak_by_shift = np.nanmax(np.abs(signed).reshape(-1, len(xs)), axis=0)

    observed = float(peak_by_shift[0])
    null = peak_by_shift[usable]
    n_use = int(usable.sum())
    return {
        'null': null,
        'observed': observed,
        'p95': float(np.percentile(null, 95)),
        'p_value': float((1.0 + (null >= observed).sum()) / (1.0 + n_use)),
        'n_shifts': n_use,
        'floor': 1.0 / (1.0 + n_use),
    }


# ── multiple testing ──────────────────────────────────────────────────────────

def bh(p_values: np.ndarray, q: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """
    Benjamini-Hochberg: control the false discovery rate across the whole sweep.

    BH compares the k-th smallest p-value against k*q/m instead, so tests sitting at the
    floor clear collectively: a hundred of them at 2.4e-4 pass a rank-100 threshold of
    2.5e-3 comfortably. What it controls is the expected share of false positives among
    the discoveries, which is the right target for a screen of this size.

    Returns (rejected, qvalues), both aligned to the input. NaN p-values are never
    rejected and take a NaN qvalue, and are excluded from m.
    """
    p = np.asarray(p_values, dtype=float)
    finite = np.isfinite(p)
    rejected = np.zeros(p.shape, dtype=bool)
    qvals = np.full(p.shape, np.nan)
    m = int(finite.sum())
    if m == 0:
        return rejected, qvals

    order = np.argsort(p[finite], kind='stable')
    ps = p[finite][order]
    ranks = np.arange(1, m + 1)

    below = ps <= ranks * q / m
    if below.any():
        k = int(np.flatnonzero(below).max()) + 1
        rej_sorted = np.zeros(m, dtype=bool)
        rej_sorted[:k] = True
    else:
        rej_sorted = np.zeros(m, dtype=bool)

    # step-up adjusted p-values, monotone from the largest rank down
    q_sorted = np.minimum.accumulate((ps * m / ranks)[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0.0, 1.0)

    inv = np.empty(m, dtype=int)
    inv[order] = np.arange(m)
    idx_finite = np.flatnonzero(finite)
    rejected[idx_finite] = rej_sorted[inv]
    qvals[idx_finite] = q_sorted[inv]
    return rejected, qvals


def verdict(rejected: bool, p_value: float, at_floor: bool) -> str:
    """
    discovery  cleared BH at the configured q -- the only claim that survives the sweep
    nominal    p <= 0.05 on its own, which a single-node view would call an edge
    noise      indistinguishable from the null

    `at_floor` records that a p-value sat at 1/(n+1), the strongest the exact test can
    report. It is not a weaker verdict -- it is the ceiling of the available evidence.
    """
    if not np.isfinite(p_value):
        return 'insufficient'
    if rejected:
        return 'discovery'
    if p_value <= 0.05:
        return 'nominal'
    return 'noise'


# ── artifact io ───────────────────────────────────────────────────────────────

def save(result: dict, path: Path, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'cell_real': result['cell_real'].astype(np.float32),
        'cell_p': result['cell_p'].astype(np.float32),
        'cell_p95': result['cell_p95'].astype(np.float32),
        'sheet_peak_real': result['sheet_peak_real'].astype(np.float32),
        'sheet_peak_p': result['sheet_peak_p'].astype(np.float64),
        'sheet_peak_p95': result['sheet_peak_p95'].astype(np.float32),
        'peak_real': result['peak_real'].astype(np.float32),
        'peak_p': result['peak_p'].astype(np.float64),
        'peak_p95': result['peak_p95'].astype(np.float32),
        'n_shifts': result['n_shifts'],
        'Δs': result['Δs'],
        'horizons': result['horizons'],
        'meta': np.array(json.dumps(meta)),
    }
    for key in ('source_bin', 'source_bin_number', 'selection_rank', 'selection_score'):
        if key in result:
            payload[key] = result[key]
    np.savez_compressed(path, **payload)


def sheet_from_node_result(result: dict, row: dict) -> dict:
    """Copy one selected bin sheet out of a full node validation result."""
    b = int(row["bin"])
    out = {
        "cell_real": result["cell_real"][:, b:b + 1, :],
        "cell_p": result["cell_p"][:, b:b + 1, :],
        "cell_p95": result["cell_p95"][:, b:b + 1, :],
        "sheet_peak_real": result["sheet_peak_real"][b:b + 1, :],
        "sheet_peak_p": result["sheet_peak_p"][b:b + 1, :],
        "sheet_peak_p95": result["sheet_peak_p95"][b:b + 1, :],
        "peak_real": result["sheet_peak_real"][b, :],
        "peak_p": result["sheet_peak_p"][b, :],
        "peak_p95": result["sheet_peak_p95"][b, :],
        "n_shifts": result["n_shifts"],
        "Δs": result["Δs"],
        "horizons": result["horizons"],
        "source_bin": np.array(b, dtype=np.int32),
        "source_bin_number": np.array(b + 1, dtype=np.int32),
        "selection_rank": np.array(int(row["rank"]), dtype=np.int32),
        "selection_score": np.array(float(row["score"]), dtype=np.float32),
        "meta": {
            **result.get("meta", {}),
            "bin_labels": [row.get("bin_label", f"bin {b + 1}")],
            "source_bin": b,
            "source_bin_number": b + 1,
            "selection_rank": int(row["rank"]),
            "selection_score": float(row["score"]),
            "selection_best_cell": row.get("best_cell"),
        },
    }
    return out


def load(path: Path) -> dict:
    z = np.load(path, allow_pickle=False)
    out = {k: z[k] for k in z.files if k != 'meta'}
    out['meta'] = json.loads(str(z['meta']))
    return out
