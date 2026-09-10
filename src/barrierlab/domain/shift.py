"""Stage 2 shift calculations: full conditional surfaces minus the baseline."""

from __future__ import annotations

import numpy as np
import torch
from barrierlab.domain import tensor_runtime
from barrierlab.domain.scoring import baseline_shifts


def from_cube(cube: dict, baseline_probability: np.ndarray) -> dict:
    """
    Convert a raw probability cube into a baseline-subtracted shift cube.

    `shift` is stored as a probability difference:

        conditional_probability - baseline_probability

    The conditional probabilities and baseline are carried too, so inspection and
    later derived artifacts can show the rate behind a shift without reloading stage 1.
    """
    conditional_probability = tensor_runtime.tensor(cube["conditional_probability"])
    baseline_probability = tensor_runtime.tensor(baseline_probability)
    if (conditional_probability.shape[0] != baseline_probability.shape[0]
            or conditional_probability.shape[2] != baseline_probability.shape[1]):
        raise ValueError(
            "baseline shape does not match cube barrier/horizon axes: "
            f"{baseline_probability.shape} vs {conditional_probability.shape}"
        )

    out = {
        "probability_shift": baseline_shifts(
            conditional_probability, baseline_probability
        ).cpu().numpy(),
        "conditional_probability": conditional_probability.cpu().numpy(),
        "baseline_probability": baseline_probability.cpu().numpy(),
        "bin_hit_counts": cube["bin_hit_counts"],
        "bin_observation_counts": cube["bin_observation_counts"],
        "eligible_observation_count": cube["eligible_observation_count"],
        "barriers": cube["barriers"],
        "horizons": cube["horizons"],
        "bin_edges": cube["bin_edges"],
        "meta": cube.get("meta", {}),
    }
    for key in ("index", "feature_values", "open", "high", "low", "close", "volume"):
        if key in cube:
            out[key] = cube[key]
    return out


def evaluate(
    cube: dict,
    min_dev: float = 0.10,
    min_bin_n: int = 50,
    min_run: int = 2,
) -> dict:
    """
    Economic filter over a shift cube.

    A node passes when at least one bin/horizon has `min_run` adjacent barrier rows with
    the same-signed deviation from baseline, each at least `min_dev` in probability units.
    """
    dev = np.asarray(cube["probability_shift"], dtype=float)
    conditional_probability = np.asarray(cube["conditional_probability"], dtype=float)
    baseline_probability = np.asarray(cube["baseline_probability"], dtype=float)
    bin_observation_counts = cube["bin_observation_counts"]
    barriers, horizons = cube["barriers"], cube["horizons"]
    barrier_count, effective_bin_count, horizon_count = dev.shape

    per_t, best_overall = {}, None
    for j in range(horizon_count):
        t = int(horizons[j])
        best = None
        for b in range(effective_bin_count):
            if bin_observation_counts[b, j] < min_bin_n:
                continue
            col = dev[:, b, j]
            run, start = 0, None
            for i in range(barrier_count):
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
                    k = int(start + np.argmax(np.abs(col[start:i + 1])))
                    cand = {
                        "horizon": t,
                        "bin": b,
                        "barrier": float(barriers[k]),
                        "dev": float(col[k]),
                        "run": run,
                        "conditional_probability": float(conditional_probability[k, b, j]),
                        "baseline_probability": float(baseline_probability[k, j]),
                        "bin_observation_count": int(bin_observation_counts[b, j]),
                        "bin_hit_count": int(cube["bin_hit_counts"][k, b, j]),
                    }
                    if best is None or abs(cand["dev"]) > abs(best["dev"]):
                        best = cand
        per_t[t] = best
        if best and (best_overall is None or abs(best["dev"]) > abs(best_overall["dev"])):
            best_overall = best

    return {
        "passed": best_overall is not None,
        "best": best_overall,
        "per_horizon": per_t,
        "criteria": {"min_dev": min_dev, "min_bin_n": min_bin_n, "min_run": min_run},
    }
