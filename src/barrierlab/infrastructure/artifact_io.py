"""Typed NPZ persistence for pipeline artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


_HISTORY_KEYS = (
    "index",
    "feature_values",
    "open",
    "high",
    "low",
    "close",
    "volume",
)


def read_json(path: Path) -> dict:
    """Read a JSON artifact, returning an empty mapping when it is unavailable."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def write_json(path: Path, payload: dict) -> None:
    """Write a human-readable JSON artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _history_payload(artifact: dict) -> dict:
    return {
        key: np.asarray(artifact[key], dtype=str) if key == "index" else artifact[key]
        for key in _HISTORY_KEYS
        if key in artifact
    }


def _write_npz(path: Path, payload: dict, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload, meta=np.array(json.dumps(meta)))


def _read_npz(path: Path, float64_keys: tuple[str, ...] = ()) -> dict:
    with np.load(path, allow_pickle=False) as stored:
        artifact = {
            key: stored[key]
            for key in stored.files
            if key != "meta"
        }
        artifact["meta"] = json.loads(str(stored["meta"]))
    for key in float64_keys:
        artifact[key] = artifact[key].astype(np.float64)
    return artifact


def save_surface(cube: dict, path: Path, meta: dict) -> None:
    """Write a complete Stage 1 surface cube."""
    payload = {
        "prob": cube["prob"].astype(np.float32),
        "hits": cube["hits"],
        "bin_n": cube["bin_n"],
        "n_obs": cube["n_obs"],
        "Δs": cube["Δs"],
        "horizons": cube["horizons"],
        "edges": cube["edges"],
        **_history_payload(cube),
    }
    _write_npz(path, payload, meta)


def load_surface(path: Path) -> dict:
    """Load a Stage 1 surface cube."""
    return _read_npz(path, ("prob",))


def save_shift(cube: dict, path: Path, meta: dict) -> None:
    """Write a complete Stage 2 baseline-subtracted shift cube."""
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
        **_history_payload(cube),
    }
    _write_npz(path, payload, meta)


def load_shift(path: Path) -> dict:
    """Load a Stage 2 baseline-subtracted shift cube."""
    return _read_npz(path, ("shift", "prob", "base"))


def save_selected_node(cube: dict, path: Path, meta: dict) -> None:
    """Write one complete Stage 3 selected-node artifact."""
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
        **_history_payload(cube),
    }
    _write_npz(path, payload, meta)


def load_selected_node(path: Path) -> dict:
    """Load and validate one complete Stage 3 selected-node artifact."""
    artifact = _read_npz(path, ("shift", "prob", "base"))
    expected_bins = len(artifact["edges"]) + 1
    expected_shape = (
        len(artifact["Δs"]),
        expected_bins,
        len(artifact["horizons"]),
    )
    source_bin = int(np.asarray(artifact.get("source_bin", -1)))
    if (
        artifact["shift"].shape != expected_shape
        or artifact["prob"].shape != expected_shape
        or artifact["hits"].shape != expected_shape
        or artifact["bin_n"].shape != expected_shape[1:]
        or artifact["base"].shape != (expected_shape[0], expected_shape[2])
        or not 0 <= source_bin < expected_bins
    ):
        raise ValueError("selected-node artifact does not contain the complete bin cube")
    return artifact


def save_validation(result: dict, path: Path, meta: dict) -> None:
    """Write one Stage 4 validation artifact."""
    payload = {
        "cell_real": result["cell_real"].astype(np.float32),
        "cell_p": result["cell_p"].astype(np.float32),
        "cell_p95": result["cell_p95"].astype(np.float32),
        "sheet_peak_real": result["sheet_peak_real"].astype(np.float32),
        "sheet_peak_p": result["sheet_peak_p"].astype(np.float64),
        "sheet_peak_p95": result["sheet_peak_p95"].astype(np.float32),
        "peak_real": result["peak_real"].astype(np.float32),
        "peak_p": result["peak_p"].astype(np.float64),
        "peak_p95": result["peak_p95"].astype(np.float32),
        "n_shifts": result["n_shifts"],
        "Δs": result["Δs"],
        "horizons": result["horizons"],
    }
    for key in ("node_peak_real", "node_peak_p", "node_peak_p95"):
        if key in result:
            dtype = np.float64 if key.endswith("_p") else np.float32
            payload[key] = np.asarray(result[key], dtype=dtype)
    for key in (
        "source_bin",
        "source_bin_number",
        "selection_rank",
        "selection_score",
    ):
        if key in result:
            payload[key] = result[key]
    _write_npz(path, payload, meta)


def load_validation(path: Path) -> dict:
    """Load one Stage 4 validation artifact."""
    return _read_npz(path)


def save_validated_bundle(path: Path, arrays: dict, meta: dict) -> None:
    """Write the Stage 4 handoff consumed exclusively by Stage 5."""
    payload = {key: np.asarray(value) for key, value in arrays.items()}
    _write_npz(path, payload, meta)


def load_validated_bundle(path: Path) -> dict:
    """Load the self-contained selected-node evidence from Stage 4."""
    return _read_npz(path)


def save_redundancy(path: Path, arrays: dict, meta: dict) -> None:
    """Write the numerical Stage 5 redundancy artifact."""
    payload = {key: np.asarray(value) for key, value in arrays.items()}
    _write_npz(path, payload, meta)


def load_redundancy(path: Path) -> dict:
    """Load the numerical Stage 5 redundancy artifact."""
    return _read_npz(path)


def save_composition_inputs(path: Path, arrays: dict, meta: dict) -> None:
    """Write Stage 5's weighted-composition handoff for Stage 6."""
    payload = {key: np.asarray(value) for key, value in arrays.items()}
    _write_npz(path, payload, meta)


def load_composition_inputs(path: Path) -> dict:
    """Load the Stage 5 handoff; Stage 6 must not reopen older stages."""
    return _read_npz(path)


def save_composition(path: Path, *, meta: dict, **arrays) -> None:
    """Write the numerical Stage 6 composition artifact."""
    payload = {key: np.asarray(value) for key, value in arrays.items()}
    _write_npz(path, payload, meta)


def load_composition(path: Path) -> dict:
    """Load the numerical Stage 6 composition artifact."""
    return _read_npz(path)
