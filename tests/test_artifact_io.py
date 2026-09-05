from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from barrierlab.infrastructure import artifact_io


def _surface_cube() -> dict:
    shape = (2, 3, 2)
    return {
        "prob": np.linspace(0.1, 0.9, np.prod(shape)).reshape(shape),
        "hits": np.arange(np.prod(shape)).reshape(shape),
        "bin_n": np.full(shape[1:], 20),
        "n_obs": np.array(60),
        "Δs": np.array([-0.10, 0.10]),
        "horizons": np.array([7, 14]),
        "edges": np.array([-0.5, 0.5]),
        "index": np.array(["2024-01-01", "2024-01-02"]),
        "close": np.array([100.0, 101.0]),
    }


class ArtifactIoTests(unittest.TestCase):
    def test_json_round_trip_and_missing_file_fallback(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "artifact.json"
            self.assertEqual(artifact_io.read_json(path), {})

            artifact_io.write_json(path, {"stage": 3, "nodes": ["atr", "vix"]})
            loaded = artifact_io.read_json(path)

        self.assertEqual(loaded, {"stage": 3, "nodes": ["atr", "vix"]})

    def test_surface_schema_and_runtime_dtype_are_stable(self) -> None:
        cube = _surface_cube()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "surface.npz"
            artifact_io.save_surface(cube, path, {"node": "example"})

            with np.load(path, allow_pickle=False) as stored:
                self.assertEqual(
                    set(stored.files),
                    {
                        "prob",
                        "hits",
                        "bin_n",
                        "n_obs",
                        "Δs",
                        "horizons",
                        "edges",
                        "index",
                        "close",
                        "meta",
                    },
                )
                self.assertEqual(stored["prob"].dtype, np.dtype(np.float32))
                self.assertEqual(json.loads(str(stored["meta"])), {"node": "example"})

            loaded = artifact_io.load_surface(path)

        self.assertEqual(loaded["prob"].dtype, np.dtype(np.float64))
        np.testing.assert_allclose(loaded["prob"], cube["prob"], rtol=1e-6)

    def test_shift_schema_and_runtime_dtypes_are_stable(self) -> None:
        cube = {
            **_surface_cube(),
            "shift": np.full((2, 3, 2), 4.25),
            "base": np.full((2, 2), 0.4),
        }
        with TemporaryDirectory() as directory:
            path = Path(directory) / "shift.npz"
            artifact_io.save_shift(cube, path, {"stage": 2})
            loaded = artifact_io.load_shift(path)

        for key in ("shift", "prob", "base"):
            self.assertEqual(loaded[key].dtype, np.dtype(np.float64))
        self.assertEqual(loaded["meta"], {"stage": 2})

    def test_validation_optional_node_peaks_round_trip_without_pickle(self) -> None:
        shape = (2, 1, 2)
        result = {
            "cell_real": np.full(shape, 1.0),
            "cell_p": np.full(shape, 0.05),
            "cell_p95": np.full(shape, 2.0),
            "sheet_peak_real": np.full((1, 2), 1.0),
            "sheet_peak_p": np.full((1, 2), 0.05),
            "sheet_peak_p95": np.full((1, 2), 2.0),
            "peak_real": np.full(2, 1.0),
            "peak_p": np.full(2, 0.05),
            "peak_p95": np.full(2, 2.0),
            "node_peak_real": np.full(2, 1.5),
            "node_peak_p": np.full(2, 0.01),
            "node_peak_p95": np.full(2, 2.5),
            "n_shifts": np.array(100),
            "Δs": np.array([-0.10, 0.10]),
            "horizons": np.array([7, 14]),
            "source_bin": np.array(0, dtype=np.int32),
        }
        with TemporaryDirectory() as directory:
            path = Path(directory) / "validation.npz"
            artifact_io.save_validation(result, path, {"stage": 4})
            loaded = artifact_io.load_validation(path)

        self.assertEqual(loaded["meta"], {"stage": 4})
        self.assertEqual(loaded["node_peak_p"].dtype, np.dtype(np.float64))
        self.assertEqual(int(loaded["source_bin"]), 0)

    def test_composition_arrays_and_metadata_round_trip(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "composition.npz"
            artifact_io.save_composition(
                path,
                meta={"stage": 6},
                probability=np.array([0.25, 0.75]),
                node_ids=np.array(["atr", "vix"], dtype=str),
            )
            loaded = artifact_io.load_composition(path)

        np.testing.assert_allclose(loaded["probability"], [0.25, 0.75])
        np.testing.assert_array_equal(loaded["node_ids"], ["atr", "vix"])
        self.assertEqual(loaded["meta"], {"stage": 6})


if __name__ == "__main__":
    unittest.main()
