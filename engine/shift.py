"""Stage 2 shift artifacts: full conditional surfaces minus the baseline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def from_cube(cube: dict, baseline: np.ndarray) -> dict:
    """
    Convert a raw probability cube into a baseline-subtracted shift cube.

    `shift` is stored in percentage points:

        100 * (P(touch theta in h | bin) - P(touch theta in h))

    The conditional probabilities and baseline are carried too, so inspection and
    later derived artifacts can show the rate behind a shift without reloading stage 1.
    """
    prob = np.asarray(cube["prob"], dtype=float)
    base = np.asarray(baseline, dtype=float)
    if prob.shape[0] != base.shape[0] or prob.shape[2] != base.shape[1]:
        raise ValueError(
            "baseline shape does not match cube theta/horizon axes: "
            f"{base.shape} vs {prob.shape}"
        )

    out = {
        "shift": (prob - base[:, None, :]) * 100.0,
        "prob": prob,
        "base": base,
        "hits": cube["hits"],
        "bin_n": cube["bin_n"],
        "n_obs": cube["n_obs"],
        "thetas": cube["thetas"],
        "horizons": cube["horizons"],
        "edges": cube["edges"],
        "meta": cube.get("meta", {}),
    }
    for key in ("index", "feature_values", "open", "high", "low", "close", "volume"):
        if key in cube:
            out[key] = cube[key]
    return out


def evaluate(
    cube: dict,
    min_dev: float = 10.0,
    min_bin_n: int = 50,
    min_run: int = 2,
) -> dict:
    """
    Economic filter over a shift cube.

    A node passes when at least one bin/horizon has `min_run` adjacent theta rows with
    the same-signed deviation from baseline, each at least `min_dev` percentage points.
    """
    dev = np.asarray(cube["shift"], dtype=float)
    prob = np.asarray(cube["prob"], dtype=float)
    base = np.asarray(cube["base"], dtype=float)
    bin_n = cube["bin_n"]
    thetas, horizons = cube["thetas"], cube["horizons"]
    n_th, n_bins, n_h = dev.shape

    per_h, best_overall = {}, None
    for j in range(n_h):
        h = int(horizons[j])
        best = None
        for b in range(n_bins):
            if bin_n[b, j] < min_bin_n:
                continue
            col = dev[:, b, j]
            run, start = 0, None
            for i in range(n_th):
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
                        "horizon": h,
                        "bin": b,
                        "theta": float(thetas[k]),
                        "dev": float(col[k]),
                        "run": run,
                        "prob": float(prob[k, b, j]),
                        "base": float(base[k, j]),
                        "bin_n": int(bin_n[b, j]),
                        "hits": int(cube["hits"][k, b, j]),
                    }
                    if best is None or abs(cand["dev"]) > abs(best["dev"]):
                        best = cand
        per_h[h] = best
        if best and (best_overall is None or abs(best["dev"]) > abs(best_overall["dev"])):
            best_overall = best

    return {
        "passed": best_overall is not None,
        "best": best_overall,
        "per_horizon": per_h,
        "criteria": {"min_dev": min_dev, "min_bin_n": min_bin_n, "min_run": min_run},
    }


def save(cube: dict, path: Path, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "shift": cube["shift"].astype(np.float32),
        "prob": cube["prob"].astype(np.float32),
        "base": cube["base"].astype(np.float32),
        "hits": cube["hits"],
        "bin_n": cube["bin_n"],
        "n_obs": cube["n_obs"],
        "thetas": cube["thetas"],
        "horizons": cube["horizons"],
        "edges": cube["edges"],
        "meta": np.array(json.dumps(meta)),
    }
    for key in ("index", "feature_values", "open", "high", "low", "close", "volume"):
        if key in cube:
            payload[key] = np.asarray(cube[key], dtype=str) if key == "index" else cube[key]
    np.savez_compressed(path, **payload)


def load(path: Path) -> dict:
    z = np.load(path, allow_pickle=False)
    out = {k: z[k] for k in z.files if k != "meta"}
    for k in ("shift", "prob", "base"):
        out[k] = out[k].astype(np.float64)
    out["meta"] = json.loads(str(z["meta"]))
    return out
