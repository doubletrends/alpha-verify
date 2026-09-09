"""Shared baseline-relative bin scoring for selection and observed/null validation."""

from __future__ import annotations

import numpy as np
import torch

from barrierlab.domain import tensor_runtime
from barrierlab.domain.barrier import MIN_BIN_N

SCORING_VERSION = "baseline-relative-grid-v2"


def paired_barriers(deltas) -> list[tuple[float, int, int]]:
    """Unique nonzero magnitudes with exactly one positive and negative row."""
    deltas = np.asarray(deltas, dtype=float)
    pairs = []
    for magnitude in sorted({abs(float(d)) for d in deltas if abs(d) > 1e-12}):
        positive = np.flatnonzero(np.isclose(deltas, magnitude, atol=1e-12))
        negative = np.flatnonzero(np.isclose(deltas, -magnitude, atol=1e-12))
        if len(positive) == len(negative) == 1:
            pairs.append((magnitude, int(positive[0]), int(negative[0])))
    return pairs


def _tensor(value):
    return value if isinstance(value, torch.Tensor) else tensor_runtime.tensor(value)


def baseline_shifts(prob, base):
    """Percentage-point shifts; axes are (..., signed barrier, bin, horizon)."""
    return 100.0 * (_tensor(prob) - _tensor(base).unsqueeze(-2))


def score_grid(prob, base, bin_n, deltas) -> dict:
    """Score one or a batch of grids using each history's own baseline.

    Probability shape: (..., signed barrier, bin, horizon). Baselines omit
    the bin axis; counts omit the barrier axis. Invalid cells contribute zero,
    and a barrier/horizon must have at least two bins with MIN_BIN_N samples.
    Returns cell details as well as the authoritative weighted sum per bin.
    """
    shifts = baseline_shifts(prob, base)
    counts = _tensor(bin_n)
    pairs = paired_barriers(deltas)
    total = torch.zeros_like(counts[..., 0], dtype=shifts.dtype)
    cells = {}
    max_magnitude = pairs[-1][0] if pairs else 1.0
    for magnitude, positive, negative in pairs:
        signed = shifts[..., positive, :, :] - shifts[..., negative, :, :]
        usable = torch.isfinite(signed) & torch.isfinite(counts) & (counts >= MIN_BIN_N)
        usable = usable & (usable.sum(dim=-2, keepdim=True) >= 2)
        skew = torch.where(usable, signed.abs(), float("nan"))
        weight = magnitude / max_magnitude
        weighted = torch.where(usable, skew * weight, 0.0)
        total += weighted.sum(dim=-1)
        cells[magnitude] = {"signed": signed, "usable": usable, "skew": skew,
                            "weighted": weighted, "weight": weight}
    return {"total": total, "cells": cells, "shifts": shifts}
