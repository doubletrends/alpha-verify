"""Combining active conditions into one barrier-touch surface, and checking it against history.

Surfaces have ``(barrier, horizon)`` axes and observation counts ``(horizon,)``,
matching one bin's face of a Stage 1 cube.
"""

from __future__ import annotations

import numpy as np

from alphaverify.domain.barrier import MIN_BIN_N


def smoothed_probability(hit_counts, observation_counts) -> np.ndarray:
    """Laplace-smoothed touch rate ``(hits + 1) / (n + 2)``, strictly inside (0, 1)."""
    return (np.asarray(hit_counts, dtype=float) + 1.0) / (
        np.asarray(observation_counts, dtype=float) + 2.0
    )


def _log_odds(probability: np.ndarray) -> np.ndarray:
    return np.log(probability) - np.log1p(-probability)


def naive_bayes_probability(
    baseline_hits, baseline_counts, condition_hits, condition_counts, min_n: int = MIN_BIN_N,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-cell naive Bayes combination of conditions in log-odds form.

        logit P(touch | conditions) = logit P_baseline + sum_k (logit P_k - logit P_baseline)

    Each condition contributes its odds ratio against the unconditional baseline,
    as if conditions were independent given the outcome. Correlated conditions
    therefore overstate the combined shift. Rates are smoothed before taking
    log-odds. A cell's support is its weakest contributor: the fewest observations
    at that horizon across the baseline and every condition. Returns the
    ``(barrier, horizon)`` probability, NaN below ``min_n`` support, and the
    ``(horizon,)`` support counts. With no conditions the result is the smoothed baseline.
    """
    baseline_hits = np.asarray(baseline_hits, dtype=float)
    baseline_counts = np.asarray(baseline_counts, dtype=float)
    if baseline_hits.ndim != 2 or baseline_counts.shape != baseline_hits.shape[1:]:
        raise ValueError("baseline hits must be (barrier, horizon) with (horizon,) counts")
    baseline = _log_odds(smoothed_probability(baseline_hits, baseline_counts[None, :]))
    log_odds = baseline.copy()
    support = baseline_counts.copy()
    for hits, counts in zip(condition_hits, condition_counts, strict=True):
        hits = np.asarray(hits, dtype=float)
        counts = np.asarray(counts, dtype=float)
        if hits.shape != baseline_hits.shape or counts.shape != baseline_counts.shape:
            raise ValueError("condition surfaces must match the baseline axes")
        log_odds += _log_odds(smoothed_probability(hits, counts[None, :])) - baseline
        support = np.minimum(support, counts)
    probability = np.where(support[None, :] >= min_n, 1.0 / (1.0 + np.exp(-log_odds)), np.nan)
    return probability, support


def joint_touch_rate(
    condition_holds, touch_mask, price_eligible, min_n: int = MIN_BIN_N,
) -> tuple[np.ndarray, np.ndarray]:
    """Historical touch rate on the bars where every condition held at once.

    ``condition_holds`` is ``(time,)``; ``touch_mask`` is the shared
    ``(horizon, time, barrier)`` matrix and ``price_eligible`` its
    ``(horizon, time)`` forward-window support. Returns the ``(barrier, horizon)``
    rate, NaN below ``min_n`` joint observations, and the ``(horizon,)`` counts.
    This is the same unsmoothed estimate Stage 1 reports for a single condition.
    """
    holds = np.asarray(condition_holds, dtype=bool)
    touch_mask = np.asarray(touch_mask, dtype=bool)
    eligible = np.asarray(price_eligible, dtype=bool) & holds[None, :]
    if touch_mask.shape[:2] != eligible.shape:
        raise ValueError("touches, eligibility, and condition must share horizon and time axes")
    counts = eligible.sum(axis=1)
    hits = (touch_mask & eligible[:, :, None]).sum(axis=1).T
    rate = np.where(counts[None, :] >= min_n, hits / np.maximum(counts[None, :], 1), np.nan)
    return rate, counts


def nesting_violations(probability, barriers, horizons, tolerance: float = 1e-12) -> int:
    """Adjacent cells that break the ordering implied by nested touch events.

    Reaching a farther barrier implies reaching a nearer one on the same side, and a
    longer horizon contains a shorter one. Probability must not rise with barrier
    distance or fall with horizon. Unsupported (NaN) cells are ignored.
    """
    probability = np.asarray(probability, dtype=float)
    barriers = np.asarray(barriers, dtype=float)
    probability = probability[np.argsort(barriers)][:, np.argsort(np.asarray(horizons))]
    ordered = np.sort(barriers)
    downside, upside = probability[ordered < 0], probability[ordered >= 0]
    with np.errstate(invalid="ignore"):
        return int(
            (np.diff(downside, axis=0) < -tolerance).sum()
            + (np.diff(upside, axis=0) > tolerance).sum()
            + (np.diff(probability, axis=1) < -tolerance).sum()
        )
