"""Stage 3 selection: rank predictors by their strongest conditional bin."""

from __future__ import annotations

import numpy as np
import torch
from barrierlab.domain import tensor_runtime


def bin_information(
    cube: dict,
    delta: float,
    horizon: int,
) -> dict | None:
    """
    Score one condition bin by its conditional-probability skew.

    Stage 2 already stores the baseline-relative shifts for every signed barrier,
    bin, and horizon. For an equal-magnitude positive and negative barrier, each
    bin's score is abs(shift(+Δ) - shift(-Δ)). Every bin competes globally;
    a node is only the identifier of the bin's condition, never a scored aggregate.
    """
    deltas = np.asarray(cube["Δs"], dtype=float)
    horizons = np.asarray(cube["horizons"], dtype=int)
    magnitude = abs(float(delta))
    positive_hits = np.flatnonzero(np.isclose(deltas, magnitude, atol=1e-12))
    negative_hits = np.flatnonzero(np.isclose(deltas, -magnitude, atol=1e-12))
    horizon_hits = np.flatnonzero(horizons == horizon)
    if len(positive_hits) != 1 or len(negative_hits) != 1 or len(horizon_hits) != 1:
        return None

    positive_i, negative_i = int(positive_hits[0]), int(negative_hits[0])
    j = int(horizon_hits[0])
    n = tensor_runtime.tensor(cube["bin_n"])[:, j]
    positive_shift = tensor_runtime.tensor(cube["shift"])[positive_i, :, j]
    negative_shift = tensor_runtime.tensor(cube["shift"])[negative_i, :, j]
    usable = torch.isfinite(n) & torch.isfinite(positive_shift) & torch.isfinite(negative_shift) & (n > 0)
    if int(usable.sum()) < 2:
        return None

    signed_skew = positive_shift - negative_shift
    bin_score = signed_skew.abs()
    best_bin = int(torch.argmax(torch.nan_to_num(bin_score, nan=float("-inf"))))
    positive_prob = np.asarray(cube["prob"], dtype=float)[positive_i, best_bin, j]
    negative_prob = np.asarray(cube["prob"], dtype=float)[negative_i, best_bin, j]
    positive_base = float(np.asarray(cube["base"], dtype=float)[positive_i, j])
    negative_base = float(np.asarray(cube["base"], dtype=float)[negative_i, j])

    labels = cube.get("meta", {}).get("bin_labels", [])
    return {
        "score": float(bin_score[best_bin] / 100.0),
        "score_pp": float(bin_score[best_bin]),
        "bin": best_bin,
        "bin_number": best_bin + 1,
        "bin_label": labels[best_bin] if best_bin < len(labels) else f"bin {best_bin + 1}",
        "bin_score": [float(value / 100.0) for value in bin_score.cpu()],
        "bin_cells": [
            {"bin": int(index), "Δ": magnitude, "Δ_abs": magnitude,
             "horizon": int(horizon), "dev": float(signed_skew[index]),
             "positive_shift": float(positive_shift[index]),
             "negative_shift": float(negative_shift[index]),
             "bin_n": int(n[index]), "score": float(bin_score[index] / 100.0),
             "score_pp": float(bin_score[index])}
            for index in torch.nonzero(usable, as_tuple=False).flatten().cpu().tolist()
        ],
        "best_cell": {
            "bin": best_bin,
            "Δ": magnitude,
            "Δ_abs": magnitude,
            "horizon": int(horizon),
            "dev": float(signed_skew[best_bin]),
            "positive_shift": float(positive_shift[best_bin]),
            "negative_shift": float(negative_shift[best_bin]),
            "positive_prob": float(positive_prob),
            "positive_base": positive_base,
            "negative_prob": float(negative_prob),
            "negative_base": negative_base,
            "bin_n": int(n[best_bin]),
            "positive_hits": int(cube["hits"][positive_i, best_bin, j]),
            "negative_hits": int(cube["hits"][negative_i, best_bin, j]),
            "score": float(bin_score[best_bin] / 100.0),
            "score_pp": float(bin_score[best_bin]),
        },
    }


def rank_nodes(
    nodes: list[dict],
    load_cube,
    top_k: int,
    delta: float,
    horizon: int,
) -> dict:
    """Return globally ranked individual condition-bin scores."""
    candidates = []
    for node in nodes:
        cube = load_cube(node)
        n_bins = int(cube["shift"].shape[1])
        if n_bins < 2:
            continue
        row = bin_information(cube, delta, horizon)
        if row is None:
            continue
        labels = cube.get("meta", {}).get("bin_labels", [])
        for cell in row["bin_cells"]:
            b = int(cell["bin"])
            candidates.append({
                "node": node["id"], "family": node["family"],
                "feature": node["feature"], "params": node["params"],
                "score": cell["score"], "score_pp": cell["score_pp"],
                "bin": b, "bin_number": b + 1,
                "bin_label": labels[b] if b < len(labels) else f"bin {b + 1}",
                "best_cell": cell,
            })

    candidates.sort(key=lambda r: (-r["score"], r["family"], r["node"], r["bin"]))
    for rank, row in enumerate(candidates, 1):
        row["rank"] = rank

    return {
        "method": {
            "score": "abs(02_shift(+Δ, bin) - 02_shift(-Δ, bin))",
            "unit": "percentage points",
            "target": {"Δ_abs": abs(float(delta)), "horizon": horizon},
            "source": "02_shift.shift",
            "selection_unit": "individual condition bin",
            "top_k": top_k,
        },
        "selected": candidates[:top_k],
        "candidates": candidates,
    }


def selected_node_from_shift_cube(cube: dict, row: dict) -> dict:
    """Promote a complete selected-node cube and attach its representative bin."""
    b = int(row["bin"])
    if not 0 <= b < cube["shift"].shape[1]:
        raise ValueError(f"representative bin {b} is outside the selected node cube")
    out = {
        "shift": cube["shift"],
        "prob": cube["prob"],
        "base": cube["base"],
        "hits": cube["hits"],
        "bin_n": cube["bin_n"],
        "n_obs": cube["n_obs"],
        "Δs": cube["Δs"],
        "horizons": cube["horizons"],
        "edges": cube["edges"],
        "source_bin": np.array(b, dtype=np.int32),
        "source_bin_number": np.array(b + 1, dtype=np.int32),
        "selection_rank": np.array(int(row["rank"]), dtype=np.int32),
        "selection_score": np.array(float(row["score"]), dtype=np.float32),
        "meta": {
            **cube.get("meta", {}),
            "source_bin": b,
            "source_bin_number": b + 1,
            "selection_rank": int(row["rank"]),
            "selection_score": float(row["score"]),
            "selection_best_cell": row.get("best_cell"),
        },
    }
    for key in ("index", "feature_values", "open", "high", "low", "close", "volume"):
        if key in cube:
            out[key] = cube[key]
    return out
