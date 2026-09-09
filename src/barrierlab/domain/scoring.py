"""In-memory full-grid bin scoring; cell contributions stay private."""

from __future__ import annotations

import numpy as np
import torch

from barrierlab.domain import tensor_runtime
from barrierlab.domain.barrier import MIN_BIN_N
from barrierlab.domain.notation import BinScoreResult

SCORING_VERSION = "baseline-relative-bin-v3"


def _paired_barriers(barriers) -> list[tuple[float, int, int]]:
    """Unique nonzero magnitudes with exactly one positive and negative row."""
    barriers = np.asarray(barriers, dtype=float)
    pairs = []
    for magnitude in sorted({abs(float(value)) for value in barriers if abs(value) > 1e-12}):
        positive = np.flatnonzero(np.isclose(barriers, magnitude, atol=1e-12))
        negative = np.flatnonzero(np.isclose(barriers, -magnitude, atol=1e-12))
        if len(positive) == len(negative) == 1:
            pairs.append((magnitude, int(positive[0]), int(negative[0])))
    return pairs


def _tensor(value):
    return value if isinstance(value, torch.Tensor) else tensor_runtime.tensor(value)


def baseline_shifts(conditional_probability, baseline_probability):
    """Percentage-point shifts; axes are (..., signed barrier, bin, horizon)."""
    return 100.0 * (
        _tensor(conditional_probability) - _tensor(baseline_probability).unsqueeze(-2)
    )


def bin_scores(conditional_probability, baseline_probability,
               bin_observation_counts, barriers) -> BinScoreResult:
    """Return bin-level scores and validity, both shaped (..., bin).

    Probabilities have axes (..., signed barrier, bin, horizon); baselines
    omit bin and counts omit barrier. All paired cell contributions are
    computed together and reduced internally. Invalid contributions are zero;
    validity distinguishes unsupported bins from supported zero-score bins.
    This function performs no I/O and exposes no individual cell operations.
    """
    conditional_probability = _tensor(conditional_probability)
    baseline_probability = _tensor(baseline_probability)
    bin_observation_counts = _tensor(bin_observation_counts)
    if (conditional_probability.ndim < 3
            or conditional_probability.shape[-3] != len(barriers)
            or baseline_probability.shape != (
                conditional_probability.shape[:-2] + conditional_probability.shape[-1:]
            )
            or bin_observation_counts.shape != (
                conditional_probability.shape[:-3] + conditional_probability.shape[-2:]
            )):
        raise ValueError("probability, baseline, counts, and barrier axes must match")
    pairs = _paired_barriers(barriers)
    if not pairs:
        shape = bin_observation_counts.shape[:-1]
        return BinScoreResult(
            bin_score=conditional_probability.new_zeros(shape),
            score_supported=torch.zeros(
                shape, dtype=torch.bool, device=conditional_probability.device
            ),
        )
    probability_shift_pp = baseline_shifts(
        conditional_probability, baseline_probability
    )
    positive = [pair[1] for pair in pairs]
    negative = [pair[2] for pair in pairs]
    signed_pair_difference = (
        probability_shift_pp[..., positive, :, :]
        - probability_shift_pp[..., negative, :, :]
    )
    cell_usable = (
        torch.isfinite(signed_pair_difference)
        & torch.isfinite(bin_observation_counts.unsqueeze(-3))
        & (bin_observation_counts.unsqueeze(-3) >= MIN_BIN_N)
    )
    cell_usable &= cell_usable.sum(dim=-2, keepdim=True) >= 2
    barrier_weights = conditional_probability.new_tensor(
        [pair[0] / pairs[-1][0] for pair in pairs]
    )[:, None, None]
    cell_contribution = torch.where(
        cell_usable, signed_pair_difference.abs() * barrier_weights, 0.0
    )
    return BinScoreResult(
        bin_score=cell_contribution.sum(dim=(-3, -1)),
        score_supported=cell_usable.any(dim=-3).any(dim=-1),
    )
