"""Stage 3 selection: rank whole predictors by their conditional information."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np


_EPS = 1e-9


def _bernoulli_kl(p: np.ndarray, q: float) -> np.ndarray:
    """KL(Bernoulli(p) || Bernoulli(q)), in nats."""
    p = np.clip(np.asarray(p, dtype=float), _EPS, 1.0 - _EPS)
    q = float(np.clip(q, _EPS, 1.0 - _EPS))
    return p * np.log(p / q) + (1.0 - p) * np.log((1.0 - p) / (1.0 - q))


def score_sheet(cube: dict, bin_index: int, min_bin_n: int) -> dict | None:
    """
    Score one 2D shift sheet by the strongest symmetric barrier skew.

    For each positive Δ and horizon, compare the baseline-subtracted upside shift
    with the matching downside shift:

        skew = shift(+Δ, t) - shift(-Δ, t)

    The sheet score is max(abs(skew)) over cells whose bin has enough observations.
    """
    dev = np.asarray(cube["shift"], dtype=float)
    Δs = np.asarray(cube["Δs"], dtype=float)
    horizons = np.asarray(cube["horizons"], dtype=int)
    bin_n = np.asarray(cube["bin_n"])
    labels = cube["meta"].get("bin_labels", [])

    pos = np.flatnonzero(Δs > 1e-12)
    pairs = []
    for i_pos in pos:
        hits = np.flatnonzero(np.isclose(Δs, -Δs[i_pos], atol=1e-12))
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
    signed_Δ = float(Δs[i_pos] if abs(up_shift) >= abs(down_shift) else Δs[i_neg])
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
            "Δ": signed_Δ,
            "Δ_abs": float(Δs[i_pos]),
            "horizon": int(horizons[j_best]),
            "dev": float(signed_dev),
            "skew": float(skew[r_best, j_best]),
            "up_shift": up_shift,
            "down_shift": down_shift,
            "bin_n": int(bin_n[bin_index, j_best]),
        },
    }


def score_node_information(
    cube: dict,
    delta: float,
    horizon: int,
    shrink_k: float,
) -> dict | None:
    """
    Score one node's complete conditional table for a Bayes target.

    The score is the sample-weighted KL divergence between the ten conditional
    Bernoulli rates and the unconditional rate.  It is also the expected log-loss
    improvement (in nats per observation) from knowing the node's bin.  Consequently
    a dramatic decile receives credit only for the observations it covers, while a
    small but persistent gradient accumulates credit across its bins.
    """
    deltas = np.asarray(cube["Δs"], dtype=float)
    horizons = np.asarray(cube["horizons"], dtype=int)
    delta_hits = np.flatnonzero(np.isclose(deltas, delta, atol=1e-12))
    horizon_hits = np.flatnonzero(horizons == horizon)
    if len(delta_hits) != 1 or len(horizon_hits) != 1:
        return None

    i, j = int(delta_hits[0]), int(horizon_hits[0])
    n = np.asarray(cube["bin_n"], dtype=float)[:, j]
    hits = np.asarray(cube["hits"], dtype=float)[i, :, j]
    prior = float(np.asarray(cube["base"], dtype=float)[i, j])
    usable = np.isfinite(n) & np.isfinite(hits) & (n > 0)
    if usable.sum() < 2 or not np.isfinite(prior):
        return None

    weights = np.zeros_like(n, dtype=float)
    weights[usable] = n[usable] / n[usable].sum()
    rates = np.full_like(n, np.nan, dtype=float)
    rates[usable] = (hits[usable] + shrink_k * prior) / (n[usable] + shrink_k)
    contribution = np.zeros_like(n, dtype=float)
    contribution[usable] = weights[usable] * _bernoulli_kl(rates[usable], prior)
    best_bin = int(np.nanargmax(contribution))

    positive = contribution[contribution > 0]
    if len(positive):
        share = positive / positive.sum()
        effective_bins = float(np.exp(-np.sum(share * np.log(share))))
    else:
        effective_bins = 0.0

    labels = cube.get("meta", {}).get("bin_labels", [])
    return {
        "score": float(contribution.sum()),
        "score_bits": float(contribution.sum() / np.log(2.0)),
        "effective_bins": effective_bins,
        "bin": best_bin,
        "bin_number": best_bin + 1,
        "bin_label": labels[best_bin] if best_bin < len(labels) else f"bin {best_bin + 1}",
        "bin_information": [float(value) for value in contribution],
        "best_cell": {
            "bin": best_bin,
            "Δ": float(delta),
            "Δ_abs": abs(float(delta)),
            "horizon": int(horizon),
            "dev": float((rates[best_bin] - prior) * 100.0),
            "prob": float(rates[best_bin]),
            "base": prior,
            "bin_n": int(n[best_bin]),
            "hits": int(hits[best_bin]),
            "information": float(contribution[best_bin]),
            "information_share": float(contribution[best_bin] / contribution.sum())
            if contribution.sum() else 0.0,
        },
    }


def rank_nodes(
    nodes: list[dict],
    load_cube,
    top_k: int,
    delta: float,
    horizon: int,
    shrink_k: float,
) -> dict:
    """Return one full-table information score and representative bin per node."""
    candidates = []
    for node in nodes:
        cube = load_cube(node)
        n_bins = int(cube["shift"].shape[1])
        if n_bins < 2:
            continue
        row = score_node_information(cube, delta, horizon, shrink_k)
        if row is None:
            continue
        b = int(row["bin"])
        candidates.append({
            "node": node["id"],
            "family": node["family"],
            "feature": node["feature"],
            "params": node["params"],
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
            "score": "sample-weighted KL(Bernoulli(P(touch | bin)) || Bernoulli(P(touch))) across all bins",
            "unit": "nats per observation",
            "target": {"Δ": delta, "horizon": horizon},
            "shrinkage_k": shrink_k,
            "representative_bin": "largest per-bin contribution to the node information score",
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
        "Δs": cube["Δs"],
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
        "Δs": cube["Δs"],
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
