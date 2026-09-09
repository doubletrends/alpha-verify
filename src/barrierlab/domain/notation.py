"""Canonical mathematical vocabulary and tensor-axis contracts.

Names in this module are deliberately descriptive and ASCII-only.  Short
symbols belong in the mathematical specification; arrays crossing a Python
API boundary use the names defined here.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import torch


class OhlcvComponent(IntEnum):
    """Positions on the final axis of an OHLCV history tensor."""

    OPEN = 0
    HIGH = 1
    LOW = 2
    CLOSE = 3
    VOLUME = 4


AXIS_CONTRACTS = {
    "ohlcv": ("replicate", "observation", "ohlcv_component"),
    "feature_values": ("replicate_or_one", "observation"),
    "bin_edges": ("replicate_or_one", "edge"),
    "bin_assignments": ("replicate", "observation"),
    "excursion": ("replicate", "observation"),
    "touch_mask": ("replicate", "observation", "barrier"),
    "conditional_probability": ("replicate", "barrier", "bin", "horizon"),
    "baseline_probability": ("replicate", "barrier", "horizon"),
    "bin_observation_counts": ("replicate", "bin", "horizon"),
    "bin_hit_counts": ("replicate", "barrier", "bin", "horizon"),
    "bin_score": ("replicate", "bin"),
}


@dataclass(frozen=True)
class MeasurementSlice:
    """One horizon of measured probabilities and counts.

    The singleton final dimension is the streamed horizon axis.
    """

    conditional_probability: torch.Tensor
    baseline_probability: torch.Tensor
    bin_hit_counts: torch.Tensor
    bin_observation_counts: torch.Tensor
    eligible_observation_count: torch.Tensor


@dataclass(frozen=True)
class BinScoreResult:
    """Full-grid scores and whether each score has supporting cells."""

    bin_score: torch.Tensor
    score_supported: torch.Tensor


@dataclass(frozen=True)
class HistoryScoreResult:
    """Scores, support flags, and effective edges for a history batch."""

    bin_score: object
    score_supported: object
    bin_edges: object
