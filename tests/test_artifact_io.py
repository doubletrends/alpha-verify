from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from safetensors import safe_open

from barrierlab.infrastructure import artifact_io
from barrierlab.pipeline.context import materialized_shift


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
            path = Path(directory) / "surface.safetensors"
            artifact_io.save_surface(cube, path, {"node": "example"})

            with safe_open(path, framework="np") as stored:
                self.assertEqual(
                    set(stored.keys()),
                    {
                        "prob",
                        "hits",
                        "bin_n",
                        "Δs",
                        "horizons",
                        "edges",
                        "close",
                    },
                )
                self.assertEqual(stored.get_tensor("prob").dtype, np.dtype(np.float32))
                self.assertEqual(json.loads(stored.metadata()["meta"]), {"node": "example"})

            loaded = artifact_io.load_surface(path)

        self.assertEqual(loaded["prob"].dtype, np.dtype(np.float64))
        np.testing.assert_allclose(loaded["prob"], cube["prob"], rtol=1e-6)
        np.testing.assert_array_equal(loaded["index"], cube["index"])

    def test_shift_schema_and_runtime_dtypes_are_stable(self) -> None:
        cube = {
            **_surface_cube(),
            "shift": np.full((2, 3, 2), 4.25),
            "base": np.full((2, 2), 0.4),
        }
        with TemporaryDirectory() as directory:
            path = Path(directory) / "shift.safetensors"
            artifact_io.save_shift(cube, path, {"stage": 2})
            loaded = artifact_io.load_shift(path)

        for key in ("shift", "prob", "base"):
            self.assertEqual(loaded[key].dtype, np.dtype(np.float64))
        self.assertEqual(loaded["meta"], {"stage": 2})

    def test_thin_shift_owns_only_the_derived_shift(self) -> None:
        cube = {**_surface_cube(), "shift": np.full((2, 3, 2), 4.25)}
        with TemporaryDirectory() as directory:
            path = Path(directory) / "shift.safetensors"
            artifact_io.save_shift(
                cube, path, {"source_artifact": "01_surface/array/a.safetensors"}, thin=True,
            )
            with safe_open(path, framework="np") as stored:
                self.assertEqual(list(stored.keys()), ["shift"])
            loaded = artifact_io.load_shift(path)
        self.assertEqual(set(loaded), {"shift", "meta"})

    def test_thin_shift_materializes_from_stage1(self) -> None:
        class WorkspaceStub:
            def __init__(self, root):
                self.root = Path(root)
                self.baseline_cube = self.cube_path("baseline")
            def cube_path(self, node):
                return self.root / "01_surface" / f"{node}.safetensors"
            def shift_cube_path(self, node):
                return self.root / "02_shift" / f"{node}.safetensors"
        with TemporaryDirectory() as directory:
            ws = WorkspaceStub(directory)
            node, baseline = _surface_cube(), _surface_cube()
            baseline["prob"] = np.broadcast_to(
                np.full((2, 1, 2), .4), (2, 3, 2),
            ).copy()
            artifact_io.save_surface(node, ws.cube_path("a"), {"node": "a"})
            artifact_io.save_surface(baseline, ws.baseline_cube, {"node": "baseline"})
            artifact_io.save_shift(
                {"shift": np.full((2, 3, 2), 4.25)}, ws.shift_cube_path("a"),
                {"source_artifact": "01_surface/a.safetensors"}, thin=True,
            )
            loaded = materialized_shift(ws, "a")
        np.testing.assert_array_equal(loaded["shift"], 4.25)
        np.testing.assert_allclose(loaded["base"], .4)
        self.assertIn("prob", loaded)


if __name__ == "__main__":
    unittest.main()
