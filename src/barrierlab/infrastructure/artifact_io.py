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

ARTIFACT_SCHEMA_VERSION = 2

# Readers accept version-1 field names so existing workspaces remain usable.
_LEGACY_ARRAY_NAMES = {
    "prob": "conditional_probability",
    "base": "baseline_probability",
    "baseline": "baseline_probability",
    "shift": "probability_shift_pp",
    "hits": "bin_hit_counts",
    "bin_n": "bin_observation_counts",
    "n_obs": "eligible_observation_count",
    "Δs": "barriers",
    "edges": "bin_edges",
    "bin_indices": "bin_assignments",
    "forward_low": "downside_excursion",
    "forward_high": "upside_excursion",
    "touches": "touch_mask",
}


def _array_value(artifact: dict, canonical_name: str):
    """Read a canonical field or its version-1 spelling from caller input."""
    if canonical_name in artifact:
        return artifact[canonical_name]
    for legacy_name, current_name in _LEGACY_ARRAY_NAMES.items():
        if current_name == canonical_name and legacy_name in artifact:
            return artifact[legacy_name]
    raise KeyError(canonical_name)


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
                "meta": json.dumps({**meta, "artifact_schema_version": ARTIFACT_SCHEMA_VERSION}),
                "encoded_arrays": json.dumps(encoded),
            },
        )
        return
    np.savez_compressed(
        path, **payload,
        meta=np.array(json.dumps({**meta, "artifact_schema_version": ARTIFACT_SCHEMA_VERSION})),
    )


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
    artifact = {_LEGACY_ARRAY_NAMES.get(key, key): value for key, value in artifact.items()}
    for key in float64_keys:
        if key in artifact:
            artifact[key] = artifact[key].astype(np.float64)
    return artifact


def save_surface(cube: dict, path: Path, meta: dict) -> None:
    """Write a complete Stage 1 surface cube."""
    payload = {
        "conditional_probability": _array_value(cube, "conditional_probability").astype(np.float32),
        "bin_hit_counts": _array_value(cube, "bin_hit_counts"),
        "bin_observation_counts": _array_value(cube, "bin_observation_counts"),
        "eligible_observation_count": _array_value(cube, "eligible_observation_count"),
        "barriers": _array_value(cube, "barriers"),
        "horizons": cube["horizons"],
        "bin_edges": _array_value(cube, "bin_edges"),
        **({"bin_assignments": _array_value(cube, "bin_assignments")}
           if "bin_assignments" in cube or "bin_indices" in cube else {}),
        **_history_payload(cube),
    }
    _write_npz(path, payload, meta)


def load_surface(path: Path) -> dict:
    """Load a Stage 1 surface cube."""
    return _read_npz(path, ("conditional_probability",))


def save_observed_cache(cache: dict, path: Path, meta: dict) -> None:
    """Persist source-level observed outcomes shared by every condition node."""
    _write_npz(path, cache, meta)


def load_observed_cache(path: Path) -> dict:
    return _read_npz(
        path, ("downside_excursion", "upside_excursion", "baseline_probability")
    )


def save_shift(cube: dict, path: Path, meta: dict, *, thin: bool = False) -> None:
    """Write a complete Stage 2 baseline-subtracted shift cube."""
    if thin:
        _write_npz(
            path, {"probability_shift_pp": _array_value(
                cube, "probability_shift_pp"
            ).astype(np.float32)}, meta
        )
        return
    payload = {
        "probability_shift_pp": _array_value(cube, "probability_shift_pp").astype(np.float32),
        "conditional_probability": _array_value(cube, "conditional_probability").astype(np.float32),
        "baseline_probability": _array_value(cube, "baseline_probability").astype(np.float32),
        "bin_hit_counts": _array_value(cube, "bin_hit_counts"),
        "bin_observation_counts": _array_value(cube, "bin_observation_counts"),
        "eligible_observation_count": _array_value(cube, "eligible_observation_count"),
        "barriers": _array_value(cube, "barriers"),
        "horizons": cube["horizons"],
        "bin_edges": _array_value(cube, "bin_edges"),
        **_history_payload(cube),
    }
    _write_npz(path, payload, meta)


def load_shift(path: Path) -> dict:
    """Load a Stage 2 baseline-subtracted shift cube."""
    return _read_npz(
        path, ("probability_shift_pp", "conditional_probability", "baseline_probability")
    )
