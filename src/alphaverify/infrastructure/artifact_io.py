"""Typed array and JSON persistence for pipeline artifacts; arrays are SafeTensors."""

from __future__ import annotations

import hashlib
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

ARTIFACT_SCHEMA_VERSION = 2
SHIFT_VERSION = "probability-difference-v1"
SHIFT_UNIT = "probability_difference"

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


def file_sha256(path: Path) -> str:
    """Content fingerprint used to tie derived artifacts to their exact inputs."""
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _history_payload(artifact: dict) -> dict:
    return {
        key: np.asarray(artifact[key], dtype=str) if key == "index" else artifact[key]
        for key in _HISTORY_KEYS
        if key in artifact
    }


def _write_arrays(path: Path, payload: dict, meta: dict) -> None:
    """Persist an array artifact as SafeTensors; scalars and strings go in the header."""
    path.parent.mkdir(parents=True, exist_ok=True)
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
            "meta": json.dumps({**meta, "artifact_schema_version": ARTIFACT_SCHEMA_VERSION}),
            "encoded_arrays": json.dumps(encoded),
        },
    )


def _read_arrays(path: Path, float64_keys: tuple[str, ...]) -> dict:
    artifact = dict(load_safetensors(str(path)))
    with safe_open(str(path), framework="np") as stored:
        header = stored.metadata() or {}
    for key, encoded in json.loads(header.get("encoded_arrays", "{}")).items():
        artifact[key] = np.asarray(
            encoded["data"], dtype=np.dtype(encoded["dtype"])
        ).reshape(encoded["shape"])
    artifact["meta"] = json.loads(header.get("meta", "{}"))
    for key in float64_keys:
        if key in artifact:
            artifact[key] = artifact[key].astype(np.float64)
    return artifact


def save_surface(cube: dict, path: Path, meta: dict) -> None:
    """Write a complete Stage 1 surface cube."""
    payload = {
        "conditional_probability": cube["conditional_probability"].astype(np.float32),
        "bin_hit_counts": cube["bin_hit_counts"],
        "bin_observation_counts": cube["bin_observation_counts"],
        "eligible_observation_count": cube["eligible_observation_count"],
        "barriers": cube["barriers"],
        "horizons": cube["horizons"],
        "bin_edges": cube["bin_edges"],
        "bin_assignments": cube["bin_assignments"],
        **_history_payload(cube),
    }
    _write_arrays(path, payload, meta)


def load_surface(path: Path) -> dict:
    """Load a Stage 1 surface cube."""
    return _read_arrays(path, ("conditional_probability",))


def save_observed_cache(cache: dict, path: Path, meta: dict) -> None:
    """Persist source-level observed outcomes shared by every condition node."""
    _write_arrays(path, cache, meta)


def load_observed_cache(path: Path) -> dict:
    return _read_arrays(
        path, ("downside_excursion", "upside_excursion", "baseline_probability")
    )


def save_shift(probability_shift: np.ndarray, path: Path, meta: dict) -> None:
    """Write a Stage 2 shift: only the derived probability differences.

    Probabilities, counts, axes, and history stay in the Stage 1 artifacts that
    ``meta`` references.
    """
    _write_arrays(
        path, {"probability_shift": np.asarray(probability_shift, dtype=np.float32)},
        {**meta, "shift_version": SHIFT_VERSION, "shift_unit": SHIFT_UNIT},
    )


def load_shift(path: Path) -> dict:
    """Load probability differences, rejecting any other shift units or version."""
    cube = _read_arrays(path, ("probability_shift",))
    meta = cube["meta"]
    if ("probability_shift" not in cube or meta.get("shift_version") != SHIFT_VERSION
            or meta.get("shift_unit") != SHIFT_UNIT):
        raise ValueError("unknown shift units/version; rerun compare")
    return cube
