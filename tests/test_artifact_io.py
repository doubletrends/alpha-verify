from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from safetensors import safe_open

from alphaverify.infrastructure import artifact_io
from alphaverify.pipeline.context import materialized_shift


def _surface_cube() -> dict:
    shape = (2, 3, 2)
    return {
        "conditional_probability": np.linspace(0.1, 0.9, np.prod(shape)).reshape(shape),
        "bin_hit_counts": np.arange(np.prod(shape)).reshape(shape),
        "bin_observation_counts": np.full(shape[1:], 20),
        "eligible_observation_count": np.array(60),
        "barriers": np.array([-0.10, 0.10]),
        "horizons": np.array([7, 14]),
        "bin_edges": np.array([-0.5, 0.5]),
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
                        "conditional_probability",
                        "bin_hit_counts",
                        "bin_observation_counts",
                        "barriers",
                        "horizons",
                        "bin_edges",
                        "close",
                    },
                )
                self.assertEqual(
                    stored.get_tensor("conditional_probability").dtype, np.dtype(np.float32)
                )
                self.assertEqual(
                    json.loads(stored.metadata()["meta"]),
                    {"node": "example", "artifact_schema_version": 2},
                )

            loaded = artifact_io.load_surface(path)

        self.assertEqual(loaded["conditional_probability"].dtype, np.dtype(np.float64))
        np.testing.assert_allclose(
            loaded["conditional_probability"], cube["conditional_probability"], rtol=1e-6
        )
        np.testing.assert_array_equal(loaded["index"], cube["index"])

    def test_shift_schema_and_runtime_dtypes_are_stable(self) -> None:
        cube = {
            **_surface_cube(),
            "probability_shift": np.full((2, 3, 2), 0.0425),
            "baseline_probability": np.full((2, 2), 0.4),
        }
        with TemporaryDirectory() as directory:
            path = Path(directory) / "shift.safetensors"
            artifact_io.save_shift(cube, path, {"stage": 2})
            loaded = artifact_io.load_shift(path)

        for key in ("probability_shift", "conditional_probability", "baseline_probability"):
            self.assertEqual(loaded[key].dtype, np.dtype(np.float64))
        self.assertEqual(loaded["meta"], {"stage": 2, "artifact_schema_version": 2,
            "value": "probability_shift", "shift_unit": artifact_io.SHIFT_UNIT,
            "shift_version": artifact_io.SHIFT_VERSION})

    def test_thin_shift_owns_only_the_derived_shift(self) -> None:
        cube = {**_surface_cube(), "probability_shift": np.full((2, 3, 2), 0.0425)}
        with TemporaryDirectory() as directory:
            path = Path(directory) / "shift.safetensors"
            artifact_io.save_shift(
                cube, path, {"source_artifact": "01_surface/array/a.safetensors"}, thin=True,
            )
            with safe_open(path, framework="np") as stored:
                self.assertEqual(list(stored.keys()), ["probability_shift"])
            loaded = artifact_io.load_shift(path)
        self.assertEqual(set(loaded), {"probability_shift", "meta"})

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
            baseline["conditional_probability"] = np.broadcast_to(
                np.full((2, 1, 2), .4), (2, 3, 2),
            ).copy()
            artifact_io.save_surface(node, ws.cube_path("a"), {"node": "a"})
            artifact_io.save_surface(baseline, ws.baseline_cube, {"node": "baseline"})
            artifact_io.save_shift(
                {"probability_shift": np.full((2, 3, 2), 0.0425)}, ws.shift_cube_path("a"),
                {"source_artifact": "01_surface/a.safetensors"}, thin=True,
            )
            loaded = materialized_shift(ws, "a")
        np.testing.assert_allclose(loaded["probability_shift"], 0.0425)
        np.testing.assert_allclose(loaded["baseline_probability"], .4)
        self.assertIn("conditional_probability", loaded)


if __name__ == "__main__":
    unittest.main()
