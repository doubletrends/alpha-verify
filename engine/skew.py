"""
Stage three: the up/down asymmetry the mirrored summary grid is built for.

The summary theta ladder is mirrored magnitudes -- +theta and -theta sit at matching
distances from entry -- so that P(touch +theta) and P(touch -theta) can be read against
each other in the same column. That comparison is the *skew* of a condition, and this
module turns it from an eyeball exercise into an artifact.

Two quantities, both in percentage points, shaped (n_mag, n_bins, n_h):

  conditional skew   P(+theta | bin) - P(-theta | bin)
                     the raw asymmetry -- still carries the asset's own drift, so on
                     BTC every condition looks skewed up before you condition on it.

  excess skew        conditional skew - the baseline node's conditional skew
                     drift removed, so what is left is the asymmetry *this condition*
                     introduces. This is the quantity --validation null-tests.

Nothing here re-measures anything: it is arithmetic on the summary cube's own `prob`
array, so a skew cell and the two probability cells it came from can never disagree.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# One threshold, defined by the module that does the binning, so measurement, skew and
# validation partition the sample identically.
from engine.barrier import MIN_BIN_N


def pair_indices(thetas: np.ndarray, tol: float = 1e-9) -> list[tuple[float, int, int]]:
    """
    (magnitude, index of +magnitude, index of -magnitude) for every positive magnitude
    on the grid, ascending. theta = 0 has no pair and is dropped.

    Raises if a +m is present without its -m: the summary grid is mirrored by
    construction (Workspace.summary_thetas), and a hand-edited universe.json that breaks
    that should fail loudly rather than skew half a ladder.
    """
    th = np.asarray(thetas, dtype=float)
    out: list[tuple[float, int, int]] = []
    for mag in sorted({round(abs(float(v)), 10) for v in th if abs(v) > tol}):
        hi = np.flatnonzero(np.abs(th - mag) <= tol)
        lo = np.flatnonzero(np.abs(th + mag) <= tol)
        if not len(hi) or not len(lo):
            raise ValueError(
                f'skew needs a mirrored theta grid; magnitude {mag} is missing its '
                f'{"+" if not len(hi) else "-"}side')
        out.append((float(mag), int(hi[0]), int(lo[0])))
    return out


def skew_from_cube(cube: dict, baseline_surface: np.ndarray) -> dict:
    """
    Conditional and excess skew for one node's summary cube.

    `baseline_surface` is the baseline node's (n_theta, n_h) probability surface -- the
    unconditional P(touch theta in h), measured as an ordinary node. Its conditional
    skew is the drift term subtracted to get excess skew.
    """
    prob = cube['prob']                      # (n_theta, n_bins, n_h)
    pairs = pair_indices(cube['thetas'])
    ip = [p[1] for p in pairs]
    im = [p[2] for p in pairs]
    mags = np.array([p[0] for p in pairs], dtype=float)

    prob_up = prob[ip, :, :]                 # (n_mag, n_bins, n_h)
    prob_dn = prob[im, :, :]
    cond = (prob_up - prob_dn) * 100.0

    base = np.asarray(baseline_surface, dtype=float)
    base_cond = (base[ip, :] - base[im, :]) * 100.0        # (n_mag, n_h)
    excess = cond - base_cond[:, None, :]

    return {
        'prob_up':     prob_up,
        'prob_dn':     prob_dn,
        'cond_skew':   cond,
        'excess_skew': excess,
        'mags':        mags,
        'horizons':    np.asarray(cube['horizons'], dtype=int),
        'bin_n':       np.asarray(cube['bin_n']),
        'edges':       np.asarray(cube['edges'], dtype=float),
        # carried so the renderer works straight off this dict, the same way it does off
        # a reloaded artifact; `save` writes the richer meta passed to it
        'meta':        {'bin_labels': cube['meta']['bin_labels']},
    }


def baseline_cond_skew(baseline_cube: dict) -> np.ndarray:
    """
    (n_mag, n_h) conditional skew of the baseline node's single bin.

    This is the drift term --validation feeds into the circular-shift null: shifting the
    feature leaves the baseline untouched, so this stays fixed under the null.
    """
    surface = baseline_cube['prob'][:, 0, :]
    return skew_from_cube(baseline_cube, surface)['cond_skew'][:, 0, :]


def peak_by_horizon(result: dict, min_n: int = MIN_BIN_N) -> dict:
    """
    Per horizon, the strongest excess-skew cell over the (magnitude, bin) face.

    Returns dict keyed by int horizon -> {mag, bin, excess, cond, bin_n} or None when no
    cell in that horizon has a bin above `min_n`.
    """
    excess = result['excess_skew']
    cond   = result['cond_skew']
    bin_n  = result['bin_n']
    mags   = result['mags']
    horizons = result['horizons']
    n_mag, n_bins, n_h = excess.shape

    out: dict[int, dict | None] = {}
    for j in range(n_h):
        face = np.abs(excess[:, :, j]).copy()
        for b in range(n_bins):
            if bin_n[b, j] < min_n:
                face[:, b] = np.nan
        if not np.isfinite(face).any():
            out[int(horizons[j])] = None
            continue
        k = int(np.nanargmax(face))
        mi, bi = divmod(k, n_bins)
        out[int(horizons[j])] = {
            'mag': float(mags[mi]), 'bin': bi,
            'excess': float(excess[mi, bi, j]), 'cond': float(cond[mi, bi, j]),
            'bin_n': int(bin_n[bi, j]),
        }
    return out


# -- artifact io ---------------------------------------------------------------

def save(result: dict, path: Path, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        prob_up=result['prob_up'].astype(np.float32),
        prob_dn=result['prob_dn'].astype(np.float32),
        cond_skew=result['cond_skew'].astype(np.float32),
        excess_skew=result['excess_skew'].astype(np.float32),
        mags=result['mags'],
        horizons=result['horizons'],
        bin_n=result['bin_n'],
        edges=result['edges'],
        meta=np.array(json.dumps(meta)),
    )


def load(path: Path) -> dict:
    z = np.load(path, allow_pickle=False)
    out = {k: z[k] for k in z.files if k != 'meta'}
    for k in ('cond_skew', 'excess_skew', 'prob_up', 'prob_dn'):
        out[k] = out[k].astype(np.float64)
    out['meta'] = json.loads(str(z['meta']))
    return out
