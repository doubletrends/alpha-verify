"""Typed NPZ persistence for pipeline artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from safetensors import safe_open
from safetensors.numpy import load_file as load_safetensors
from safetensors.numpy import save_file as save_safetensors


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
    """Persist an array artifact, using SafeTensors when requested by its suffix."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".safetensors":
        tensors, encoded = {}, {}
        for key, value in payload.items():
            array = np.asarray(value)
            if array.ndim == 0 or array.dtype.kind in "OUS":
                encoded[key] = {
                    "dtype": array.dtype.str,
                    "shape": array.shape,
                    "data": array.tolist(),
                }
            else:
                tensors[key] = np.ascontiguousarray(array)
        save_safetensors(
            tensors,
            str(path),
            metadata={
                "meta": json.dumps(meta),
                "encoded_arrays": json.dumps(encoded),
            },
        )
        return
    np.savez_compressed(path, **payload, meta=np.array(json.dumps(meta)))


def _read_npz(path: Path, float64_keys: tuple[str, ...] = ()) -> dict:
    if path.suffix == ".safetensors":
        artifact = dict(load_safetensors(str(path)))
        with safe_open(str(path), framework="np") as stored:
            header = stored.metadata() or {}
        for key, encoded in json.loads(header.get("encoded_arrays", "{}")).items():
            artifact[key] = np.asarray(
                encoded["data"], dtype=np.dtype(encoded["dtype"])
            ).reshape(encoded["shape"])
        artifact["meta"] = json.loads(header.get("meta", "{}"))
    else:
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
