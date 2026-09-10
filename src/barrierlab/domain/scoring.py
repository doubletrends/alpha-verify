"""In-memory full-grid bin scoring; cell contributions stay private."""

from __future__ import annotations

import numpy as np
import torch

from barrierlab.domain import tensor_runtime
from barrierlab.domain.barrier import MIN_BIN_N
from barrierlab.domain.notation import BinScoreResult

SCORING_VERSION = "baseline-relative-bin-v5-probability-difference"


def _barrier_reflection(barriers) -> tuple[np.ndarray, np.ndarray]:
    """Full-axis mirror indices and weights; zero/unmatched rows have no weight."""
    barriers = np.asarray(barriers, dtype=float)
    mirror = np.arange(len(barriers))
    weights = np.zeros(len(barriers), dtype=float)
    largest = 0.0
    for magnitude in sorted({abs(float(value)) for value in barriers if abs(value) > 1e-12}):
        positive = np.flatnonzero(np.isclose(barriers, magnitude, atol=1e-12))
        negative = np.flatnonzero(np.isclose(barriers, -magnitude, atol=1e-12))
        if len(positive) == len(negative) == 1:
            p, n = int(positive[0]), int(negative[0])
            mirror[p], mirror[n] = n, p
            # Both signed rows carry half of the former pair contribution.
            weights[p] += magnitude / 2
            weights[n] += magnitude / 2
            largest = magnitude
    if largest:
        weights /= largest
    return mirror, weights


def _tensor(value):
    return value if isinstance(value, torch.Tensor) else tensor_runtime.tensor(value)


def baseline_shifts(conditional_probability, baseline_probability):
    """Probability differences; axes are (..., signed barrier, bin, horizon)."""
    return (
        _tensor(conditional_probability) - _tensor(baseline_probability).unsqueeze(-2)
    )


def bin_scores(conditional_probability, baseline_probability,
               bin_observation_counts, barriers) -> BinScoreResult:
    """Return bin-level scores and validity, both shaped (..., bin).

    Probabilities have axes (..., signed barrier, bin, horizon); baselines
    omit bin and counts omit barrier. Reflection, usability, and contributions
    retain the full signed-barrier axis until the final reduction. Each row
    carries half the weighted absolute difference from its reflected row.
    Invalid contributions are zero;
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
    mirror, weights = _barrier_reflection(barriers)
    if not weights.any():
        shape = bin_observation_counts.shape[:-1]
        return BinScoreResult(
            bin_score=conditional_probability.new_zeros(shape),
            score_supported=torch.zeros(
                shape, dtype=torch.bool, device=conditional_probability.device
            ),
        )
    probability_shift = baseline_shifts(
        conditional_probability, baseline_probability
    )
    mirror_indices = torch.as_tensor(
        mirror, dtype=torch.long, device=probability_shift.device,
    )
    reflected_difference = (
        probability_shift
        - probability_shift.index_select(-3, mirror_indices)
    )
    barrier_weights = conditional_probability.new_tensor(weights)[:, None, None]
    cell_usable = (
        (barrier_weights > 0)
        & torch.isfinite(reflected_difference)
        & torch.isfinite(bin_observation_counts.unsqueeze(-3))
        & (bin_observation_counts.unsqueeze(-3) >= MIN_BIN_N)
    )
    cell_usable &= cell_usable.sum(dim=-2, keepdim=True) >= 2
    cell_contribution = torch.where(
        cell_usable, reflected_difference.abs() * barrier_weights, 0.0
    )
    return BinScoreResult(
        bin_score=cell_contribution.sum(dim=(-3, -1)),
        score_supported=cell_usable.any(dim=-3).any(dim=-1),
    )
