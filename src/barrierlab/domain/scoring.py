"""In-memory full-grid bin scoring; cell contributions stay private."""

from __future__ import annotations

import numpy as np
import torch

from barrierlab.domain import tensor_runtime
from barrierlab.domain.barrier import MIN_BIN_N

SCORING_VERSION = "baseline-relative-bin-v3"


def _paired_barriers(deltas) -> list[tuple[float, int, int]]:
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


def bin_scores(prob, base, bin_n, deltas) -> dict[str, torch.Tensor]:
    """Return bin-level scores and validity, both shaped (..., bin).

    Probabilities have axes (..., signed barrier, bin, horizon); baselines
    omit bin and counts omit barrier. All paired cell contributions are
    computed together and reduced internally. Invalid contributions are zero;
    validity distinguishes unsupported bins from supported zero-score bins.
    This function performs no I/O and exposes no individual cell operations.
    """
    prob, base, counts = _tensor(prob), _tensor(base), _tensor(bin_n)
    if (prob.ndim < 3 or prob.shape[-3] != len(deltas)
            or base.shape != prob.shape[:-2] + prob.shape[-1:]
            or counts.shape != prob.shape[:-3] + prob.shape[-2:]):
        raise ValueError("probability, baseline, counts, and barrier axes must match")
    pairs = _paired_barriers(deltas)
    if not pairs:
        shape = counts.shape[:-1]
        return {"scores": prob.new_zeros(shape),
                "valid": torch.zeros(shape, dtype=torch.bool, device=prob.device)}
    shifts = baseline_shifts(prob, base)
    positive = [pair[1] for pair in pairs]
    negative = [pair[2] for pair in pairs]
    signed = shifts[..., positive, :, :] - shifts[..., negative, :, :]
    usable = (torch.isfinite(signed) & torch.isfinite(counts.unsqueeze(-3))
              & (counts.unsqueeze(-3) >= MIN_BIN_N))
    usable = usable & (usable.sum(dim=-2, keepdim=True) >= 2)
    weights = prob.new_tensor([pair[0] / pairs[-1][0] for pair in pairs])[:, None, None]
    contributions = torch.where(usable, signed.abs() * weights, 0.0)
    return {"scores": contributions.sum(dim=(-3, -1)),
            "valid": usable.any(dim=-3).any(dim=-1)}
