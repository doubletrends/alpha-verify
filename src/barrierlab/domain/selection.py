"""Stage 3 selection: rank conditional bins by their full-grid evidence."""

from __future__ import annotations

import numpy as np
import torch
from barrierlab.domain import tensor_runtime
from barrierlab.domain.scoring import SCORING_VERSION, score_grid


def bin_information(
    cube: dict,
    delta: float,
    horizon: int,
    scored: dict | None = None,
) -> dict | None:
    """
    Score one condition bin by its conditional-probability skew.

    The shared scorer derives baseline-relative shifts from the Stage 2
    probabilities and baseline. For an equal-magnitude positive and negative
    barrier, each bin's score is abs(shift(+Δ) - shift(-Δ)). Every bin competes globally;
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
    if scored is None:
        scored = score_grid(cube["prob"], cube["base"], cube["bin_n"], deltas)
    matched_magnitude = next((d for d in scored["cells"] if np.isclose(d, magnitude, atol=1e-12)), None)
    cell = scored["cells"].get(matched_magnitude)
    if cell is None:
        return None
    positive_shift = scored["shifts"][positive_i, :, j]
    negative_shift = scored["shifts"][negative_i, :, j]
    usable = cell["usable"][:, j]
    if int(usable.sum()) < 2:
        return None

    signed_skew = cell["signed"][:, j]
    bin_score = cell["skew"][:, j]
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
    progress=None,
) -> dict:
    """Return globally ranked condition-bin scores accumulated over the full grid.

    Every available two-sided barrier and horizon contributes a cell score. The
    raw skew is weighted linearly by its barrier magnitude relative to the
    largest paired magnitude available in the cube, then summed per node/bin.
    """
    candidates = []
    for node in nodes:
        try:
            cube = load_cube(node)
            n_bins = int(cube["shift"].shape[1])
            if n_bins < 2:
                continue
            deltas = np.asarray(cube["Δs"], dtype=float)
            horizons = np.asarray(cube["horizons"], dtype=int)
            scored = score_grid(cube["prob"], cube["base"], cube["bin_n"], deltas)
            paired = list(scored["cells"])
            if not paired:
                continue
            labels = cube.get("meta", {}).get("bin_labels", [])
            bin_candidates = {}
            for magnitude in paired:
                for j, horizon in enumerate(horizons):
                    row = bin_information(cube, magnitude, int(horizon), scored)
                    if row is None:
                        continue
                    weight = scored["cells"][magnitude]["weight"]
                    for cell in row["bin_cells"]:
                        b = int(cell["bin"])
                        skew_pp = cell["score_pp"]
                        score_pp = float(scored["cells"][magnitude]["weighted"][b, j])
                        scored_cell = {
                            **cell,
                            "delta_weight": weight,
                            "skew": skew_pp / 100.0,
                            "skew_pp": skew_pp,
                            "score": score_pp / 100.0,
                            "score_pp": score_pp,
                        }
                        candidate = bin_candidates.setdefault(b, {
                            "node": node["id"], "family": node["family"],
                            "feature": node["feature"], "params": node["params"],
                            "score": 0.0, "score_pp": 0.0,
                            "bin": b, "bin_number": b + 1,
                            "bin_label": labels[b] if b < len(labels) else f"bin {b + 1}",
                            "cell_count": 0, "best_cell": None,
                        })
                        candidate["cell_count"] += 1
                        if (
                            candidate["best_cell"] is None
                            or scored_cell["score"] > candidate["best_cell"]["score"]
                        ):
                            candidate["best_cell"] = scored_cell
            for candidate in bin_candidates.values():
                candidate["score_pp"] = float(scored["total"][candidate["bin"]])
                candidate["score"] = candidate["score_pp"] / 100.0
                # The representative cell drives the Stage 3 view; validation
                # and ranking both use the shared full-grid total.
                candidate["delta"] = candidate["best_cell"]["Δ"]
                candidate["delta_abs"] = candidate["best_cell"]["Δ_abs"]
                candidate["delta_weight"] = candidate["best_cell"]["delta_weight"]
                candidate["horizon"] = candidate["best_cell"]["horizon"]
                candidates.append(candidate)
        finally:
            if progress is not None:
                progress.advance()

    candidates.sort(
        key=lambda r: (-r["score"], r["family"], r["node"], r["bin"])
    )
    for rank, row in enumerate(candidates, 1):
        row["rank"] = rank

    return {
        "method": {
            "scoring_version": SCORING_VERSION,
            "cell_score": "abs(02_shift(+Δ, bin, horizon) - 02_shift(-Δ, bin, horizon)) * (abs(Δ) / max_abs_Δ)",
            "score": "sum(cell_score for all valid paired-Δ and horizon cells in a condition bin)",
            "unit": "linearly Δ-weighted percentage points",
            "selection_unit": "individual condition bin",
            "delta_weight": "abs(Δ) / max_abs_Δ within each node cube",
            "source": "02_shift.prob and 02_shift.base via scoring.score_grid",
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
        "selection_delta": np.array(float(row["delta"]), dtype=np.float32),
        "selection_horizon": np.array(int(row["horizon"]), dtype=np.int32),
        "meta": {
            **cube.get("meta", {}),
            "source_bin": b,
            "source_bin_number": b + 1,
            "selection_rank": int(row["rank"]),
            "selection_score": float(row["score"]),
            "selection_delta": float(row["delta"]),
            "selection_horizon": int(row["horizon"]),
            "selection_best_cell": row.get("best_cell"),
        },
    }
    for key in ("index", "feature_values", "open", "high", "low", "close", "volume"):
        if key in cube:
            out[key] = cube[key]
    return out
