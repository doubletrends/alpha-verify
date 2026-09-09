"""Canonical mathematical names and compatibility at persistence boundaries."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from barrierlab.domain.notation import AXIS_CONTRACTS, OhlcvComponent
from barrierlab.infrastructure import artifact_io


def test_canonical_axes_and_ohlcv_order_are_explicit() -> None:
    assert AXIS_CONTRACTS["conditional_probability"] == (
        "replicate", "barrier", "bin", "horizon",
    )
    assert AXIS_CONTRACTS["baseline_probability"] == (
        "replicate", "barrier", "horizon",
    )
    assert tuple(component.name.lower() for component in OhlcvComponent) == (
        "open", "high", "low", "close", "volume",
    )


def test_version_one_surface_names_are_normalized_on_read(tmp_path: Path) -> None:
    path = tmp_path / "legacy.safetensors"
    artifact_io._write_npz(path, {
        "prob": np.ones((2, 1, 1)),
        "hits": np.ones((2, 1, 1), dtype=np.int32),
        "bin_n": np.ones((1, 1), dtype=np.int32),
        "n_obs": np.array(1, dtype=np.int32),
        "Δs": np.array([-.1, .1]),
        "horizons": np.array([1]),
        "edges": np.array([]),
    }, {})

    loaded = artifact_io.load_surface(path)

    assert "prob" not in loaded and "Δs" not in loaded
    assert loaded["conditional_probability"].dtype == np.float64
    np.testing.assert_array_equal(loaded["barriers"], [-.1, .1])
    np.testing.assert_array_equal(loaded["bin_observation_counts"], [[1]])
