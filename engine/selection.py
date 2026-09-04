"""Stage 3 selection: rank node/bin sheets by upside-vs-downside skew."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np


def score_sheet(cube: dict, bin_index: int, min_bin_n: int) -> dict | None:
    """
    Score one 2D shift sheet by the strongest symmetric barrier skew.

    For each positive theta and horizon, compare the baseline-subtracted upside shift
    with the matching downside shift:

        skew = shift(+theta, h) - shift(-theta, h)

    The sheet score is max(abs(skew)) over cells whose bin has enough observations.
    """
    dev = np.asarray(cube["shift"], dtype=float)
    thetas = np.asarray(cube["thetas"], dtype=float)
    horizons = np.asarray(cube["horizons"], dtype=int)
    bin_n = np.asarray(cube["bin_n"])
    labels = cube["meta"].get("bin_labels", [])

    pos = np.flatnonzero(thetas > 1e-12)
    pairs = []
    for i_pos in pos:
        hits = np.flatnonzero(np.isclose(thetas, -thetas[i_pos], atol=1e-12))
        if len(hits):
            pairs.append((i_pos, int(hits[0])))
    if not pairs:
        return None

    skew = np.full((len(pairs), len(horizons)), np.nan)
    valid_h = bin_n[bin_index, :] >= min_bin_n
    for r, (i_pos, i_neg) in enumerate(pairs):
        skew[r, valid_h] = dev[i_pos, bin_index, valid_h] - dev[i_neg, bin_index, valid_h]
    if not np.isfinite(skew).any():
        return None

    r_best, j_best = np.unravel_index(int(np.nanargmax(np.abs(skew))), skew.shape)
    i_pos, i_neg = pairs[int(r_best)]
    up_shift = float(dev[i_pos, bin_index, j_best])
    down_shift = float(dev[i_neg, bin_index, j_best])
    signed_theta = float(thetas[i_pos] if abs(up_shift) >= abs(down_shift) else thetas[i_neg])
    signed_dev = up_shift if abs(up_shift) >= abs(down_shift) else down_shift

    finite = np.abs(skew[np.isfinite(skew)])
    return {
        "bin": int(bin_index),
        "bin_number": int(bin_index + 1),
        "bin_label": labels[bin_index] if bin_index < len(labels) else f"bin {bin_index + 1}",
        "score": float(finite.max()),
        "mean_abs_skew": float(finite.mean()),
        "n_scored_cells": int(finite.size),
        "best_cell": {
            "bin": int(bin_index),
            "theta": signed_theta,
            "theta_abs": float(thetas[i_pos]),
            "horizon": int(horizons[j_best]),
            "dev": float(signed_dev),
            "skew": float(skew[r_best, j_best]),
            "up_shift": up_shift,
            "down_shift": down_shift,
            "bin_n": int(bin_n[bin_index, j_best]),
        },
    }


def economic_filter_sheet(
    cube: dict,
    bin_index: int,
    min_dev: float,
    min_bin_n: int,
    min_run: int,
) -> dict:
    """Economic filter scoped to one selected node/bin sheet."""
    dev = np.asarray(cube["shift"], dtype=float)
    prob = np.asarray(cube["prob"], dtype=float)
    base = np.asarray(cube["base"], dtype=float)
    bin_n = np.asarray(cube["bin_n"])
    thetas = np.asarray(cube["thetas"], dtype=float)
    horizons = np.asarray(cube["horizons"], dtype=int)

    best = None
    for j, h in enumerate(horizons):
        if bin_n[bin_index, j] < min_bin_n:
            continue
        col = dev[:, bin_index, j]
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
                cand = {
                    "horizon": int(h),
                    "bin": int(bin_index),
                    "bin_number": int(bin_index + 1),
                    "theta": float(thetas[k]),
                    "dev": float(col[k]),
                    "run": int(run),
                    "prob": float(prob[k, bin_index, j]),
                    "base": float(base[k, j]),
                    "bin_n": int(bin_n[bin_index, j]),
                    "hits": int(cube["hits"][k, bin_index, j]),
                }
                if best is None or abs(cand["dev"]) > abs(best["dev"]):
                    best = cand

    return {
        "passed": best is not None,
        "best": best,
        "criteria": {"min_dev": min_dev, "min_bin_n": min_bin_n, "min_run": min_run},
    }


def rank_shift_sheets(
    nodes: list[dict],
    load_cube,
    min_dev: float,
    min_bin_n: int,
    min_run: int,
    top_k: int,
) -> dict:
    """Return every scored sheet plus the global top-k selection."""
    candidates = []
    for node in nodes:
        cube = load_cube(node)
        n_bins = int(cube["shift"].shape[1])
        if n_bins < 2:
            continue
        for b in range(n_bins):
            row = score_sheet(cube, b, min_bin_n)
            if row is None:
                continue
            candidates.append({
                "node": node["id"],
                "family": node["family"],
                "feature": node["feature"],
                "params": node["params"],
                "economic": economic_filter_sheet(cube, b, min_dev, min_bin_n, min_run),
                **row,
            })

    candidates.sort(key=lambda r: (-r["score"], r["family"], r["node"], r["bin"]))
    for rank, row in enumerate(candidates, 1):
        row["rank"] = rank
        row["selected"] = rank <= top_k

    return {
        "generated": datetime.now(timezone.utc).isoformat(),
        "artifact": "03_selection",
        "source": "02_shift_array",
        "method": {
            "score": "max |shift(+theta,h) - shift(-theta,h)| over theta>0 and horizons",
            "unit": "percentage points",
            "economic": "|dev| >= min_dev across adjacent same-sign theta rows in the selected bin",
            "min_dev": min_dev,
            "min_bin_n": min_bin_n,
            "min_run": min_run,
            "top_k": top_k,
        },
        "selected": candidates[:top_k],
        "candidates": candidates,
    }


def selected_bins_by_node(selection: dict) -> dict[str, set[int]]:
    """Map node id to selected zero-based bin indexes."""
    out: dict[str, set[int]] = {}
    for row in selection.get("selected", []):
        out.setdefault(row["node"], set()).add(int(row["bin"]))
    return out


def sheet_from_shift_cube(cube: dict, row: dict) -> dict:
    """Copy one selected bin sheet out of a Stage 2 shift cube."""
    b = int(row["bin"])
    out = {
        "shift": cube["shift"][:, b:b + 1, :],
        "prob": cube["prob"][:, b:b + 1, :],
        "base": cube["base"],
        "hits": cube["hits"][:, b:b + 1, :],
        "bin_n": cube["bin_n"][b:b + 1, :],
        "n_obs": cube["n_obs"],
        "thetas": cube["thetas"],
        "horizons": cube["horizons"],
        "edges": cube["edges"],
        "source_bin": np.array(b, dtype=np.int32),
        "source_bin_number": np.array(b + 1, dtype=np.int32),
        "selection_rank": np.array(int(row["rank"]), dtype=np.int32),
        "selection_score": np.array(float(row["score"]), dtype=np.float32),
        "meta": {
            **cube.get("meta", {}),
            "bin_labels": [row.get("bin_label", f"bin {b + 1}")],
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


def save_sheet(cube: dict, path: Path, meta: dict) -> None:
    """Write one selected sheet as a Stage 3 array artifact."""
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
        "source_bin": cube["source_bin"],
        "source_bin_number": cube["source_bin_number"],
        "selection_rank": cube["selection_rank"],
        "selection_score": cube["selection_score"],
        "meta": np.array(json.dumps(meta)),
    }
    for key in ("index", "feature_values", "open", "high", "low", "close", "volume"):
        if key in cube:
            payload[key] = np.asarray(cube[key], dtype=str) if key == "index" else cube[key]
    np.savez_compressed(path, **payload)


def load_sheet(path: Path) -> dict:
    """Read one selected sheet artifact."""
    z = np.load(path, allow_pickle=False)
    out = {k: z[k] for k in z.files if k != "meta"}
    for k in ("shift", "prob", "base"):
        out[k] = out[k].astype(np.float64)
    out["meta"] = json.loads(str(z["meta"]))
    return out
