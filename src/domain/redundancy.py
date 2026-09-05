"""Dependence diagnostics for discretized conditional-prediction nodes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _entropy(values: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    _, count = np.unique(values, return_counts=True)
    p = count.astype(float) / count.sum()
    return float(-np.sum(p * np.log(p)))


def _mutual_information(left: np.ndarray, right: np.ndarray) -> float:
    """Discrete mutual information in nats for aligned integer states."""
    if len(left) == 0:
        return 0.0
    _, x = np.unique(left, return_inverse=True)
    _, z = np.unique(right, return_inverse=True)
    joint = np.zeros((x.max() + 1, z.max() + 1), dtype=float)
    np.add.at(joint, (x, z), 1.0)
    joint /= joint.sum()
    px = joint.sum(axis=1, keepdims=True)
    pz = joint.sum(axis=0, keepdims=True)
    valid = joint > 0
    return float(np.sum(joint[valid] * np.log(joint[valid] / (px * pz)[valid])))


def conditional_nmi(left: np.ndarray, right: np.ndarray, y: np.ndarray) -> float:
    """
    Normalized I(left; right | y), measuring a Naive-Bayes independence violation.

    Values lie in [0, 1] up to floating-point noise. Conditioning on the barrier-touch
    label separates genuinely duplicated predictors from features that merely share a
    useful relationship with the outcome.
    """
    left = np.asarray(left, dtype=int)
    right = np.asarray(right, dtype=int)
    y = np.asarray(y, dtype=int)
    if not (len(left) == len(right) == len(y)):
        raise ValueError("conditional dependence arrays must align")
    n = len(y)
    if n == 0:
        return 0.0

    cmi = h_left = h_right = 0.0
    for label in np.unique(y):
        mask = y == label
        weight = float(mask.mean())
        cmi += weight * _mutual_information(left[mask], right[mask])
        h_left += weight * _entropy(left[mask])
        h_right += weight * _entropy(right[mask])
    denominator = float(np.sqrt(h_left * h_right))
    return 0.0 if denominator <= 1e-12 else float(np.clip(cmi / denominator, 0.0, 1.0))


def conditional_nmi_matrix(bins: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pairwise conditional-NMI matrix for rows of node bin states."""
    bins = np.asarray(bins, dtype=int)
    n_features = bins.shape[0]
    out = np.eye(n_features, dtype=float)
    for i in range(n_features):
        for j in range(i + 1, n_features):
            out[i, j] = out[j, i] = conditional_nmi(bins[i], bins[j], y)
    return out


def clusters(similarity: np.ndarray, threshold: float = 0.30) -> list[list[int]]:
    """Connected components after linking nodes with conditional NMI >= threshold."""
    similarity = np.asarray(similarity, dtype=float)
    n = similarity.shape[0]
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def join(i: int, j: int) -> None:
        left, right = find(i), find(j)
        if left != right:
            parent[right] = left

    for i in range(n):
        for j in range(i + 1, n):
            if similarity[i, j] >= threshold:
                join(i, j)

    grouped: dict[int, list[int]] = {}
    for i in range(n):
        grouped.setdefault(find(i), []).append(i)
    return sorted(grouped.values(), key=lambda group: (group[0], len(group)))


def save(path: Path, arrays: dict, meta: dict) -> None:
    """Persist the numerical Stage 5 result without pickle-dependent objects."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {key: np.asarray(value) for key, value in arrays.items()}
    payload["meta"] = np.array(json.dumps(meta))
    np.savez_compressed(path, **payload)


def load(path: Path) -> dict:
    """Load a Stage 5 numerical artifact with its JSON metadata."""
    with np.load(path, allow_pickle=False) as stored:
        out = {key: stored[key] for key in stored.files if key != "meta"}
        out["meta"] = json.loads(str(stored["meta"]))
    return out
