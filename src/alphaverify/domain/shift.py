"""Stage 2 shift calculations: full conditional surfaces minus the baseline."""

from __future__ import annotations

import numpy as np
from alphaverify.domain import tensor_runtime
from alphaverify.domain.scoring import baseline_shifts


def from_cube(cube: dict, baseline_probability: np.ndarray) -> np.ndarray:
    """
    A raw probability cube's baseline-subtracted shift, as a probability difference:

        conditional_probability - baseline_probability
    """
    conditional_probability = tensor_runtime.tensor(cube["conditional_probability"])
    baseline_probability = tensor_runtime.tensor(baseline_probability)
    if (conditional_probability.shape[0] != baseline_probability.shape[0]
            or conditional_probability.shape[2] != baseline_probability.shape[1]):
        raise ValueError(
            "baseline shape does not match cube barrier/horizon axes: "
            f"{baseline_probability.shape} vs {conditional_probability.shape}"
        )

    return baseline_shifts(conditional_probability, baseline_probability).cpu().numpy()


def strongest_cell(cube: dict, bin_index: int, min_dev: float, min_bin_n: int,
                   min_run: int) -> dict | None:
    """
    One condition bin's largest shift inside a qualifying run, or None.

    A run is at least `min_run` adjacent barrier rows with the same-signed deviation from
    baseline, each at least `min_dev` in probability units, at a horizon where the bin has
    at least `min_bin_n` observations.
    """
    dev = np.asarray(cube["probability_shift"], dtype=float)
    conditional_probability = np.asarray(cube["conditional_probability"], dtype=float)
    baseline_probability = np.asarray(cube["baseline_probability"], dtype=float)
    bin_observation_counts = cube["bin_observation_counts"]
    barriers, horizons = cube["barriers"], cube["horizons"]
    b = bin_index

    best = None
    for j, t in enumerate(horizons):
        if bin_observation_counts[b, j] < min_bin_n:
            continue
        col = dev[:, b, j]
        run, start = 0, None
        for i, v in enumerate(col):
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
                if best is None or abs(col[k]) > abs(best["dev"]):
                    best = {
                        "horizon": int(t),
                        "bin": b,
                        "barrier": float(barriers[k]),
                        "dev": float(col[k]),
                        "run": run,
                        "conditional_probability": float(conditional_probability[k, b, j]),
                        "baseline_probability": float(baseline_probability[k, j]),
                        "bin_observation_count": int(bin_observation_counts[b, j]),
                        "bin_hit_count": int(cube["bin_hit_counts"][k, b, j]),
                    }
    return best
