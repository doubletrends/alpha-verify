"""Evidence across many condition-bin tests: false discovery rate adjustment."""

from __future__ import annotations

import numpy as np


def benjamini_hochberg(p_values) -> np.ndarray:
    """Benjamini-Hochberg q-values, returned in input order.

    A bin's q-value is the smallest false discovery rate at which it would be
    accepted, when every test with an equal or smaller p-value is accepted too.
    It orders bins exactly as p does, so equal p-values share a q-value. The
    adjustment assumes independent or positively dependent tests.
    """
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1:
        raise ValueError("p-values must be a vector")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("p-values must lie in [0, 1]")
    if not len(p):
        return p.copy()
    order = np.argsort(p, kind="stable")
    # Scale by m/j (never below one) so rounding cannot put q below p.
    scaled = p[order] * (len(p) / np.arange(1, len(p) + 1))
    ordered_q = np.minimum(np.minimum.accumulate(scaled[::-1])[::-1], 1.0)
    q = np.empty_like(ordered_q)
    q[order] = ordered_q
    return q
