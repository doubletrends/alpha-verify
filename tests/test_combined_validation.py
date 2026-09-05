from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from domain.validation import economic_filter_sheet, verdict
from pipeline.step_04_validation import finalize_validation, validation_summary_is_current


class _IncompleteWorkspace:
    root_dir = Path("C:/workspace")
    selection_path = Path("C:/workspace/workspaces/example/03_selection_array/selection.json")
    validation_summary_path = Path(
        "C:/workspace/workspaces/example/04_validation_array/04_validation.json"
    )
    null_alpha = 0.01
    min_dev = 10.0
    min_bin_n = 50
    min_run = 2

    def __init__(self) -> None:
        self.dir = Path("C:/workspace/workspaces/example")
        self.written = None

    def read_json(self, path: Path) -> dict:
        return {"selected": [{"rank": 1, "node": "atr", "bin": 9, "score": 0.02}]}

    def has_selection_array(self, row: dict) -> bool:
        return True

    def write_json(self, path: Path, payload: dict) -> None:
        self.written = payload


class CombinedValidationTests(unittest.TestCase):
    def test_incomplete_null_set_cannot_publish_final_verdicts(self) -> None:
        workspace = _IncompleteWorkspace()
        with patch("pipeline.step_04_validation._validation_matches_selection", return_value=False):
            complete = finalize_validation(workspace)

        self.assertFalse(complete)
        self.assertFalse(workspace.written["complete"])
        self.assertEqual(workspace.written["missing_nodes"], ["atr"])
        self.assertEqual(workspace.written["tests"], [])
        self.assertEqual(workspace.written["cleared"], [])

    def test_stale_selection_fingerprint_is_rejected(self) -> None:
        workspace = _IncompleteWorkspace()
        summary = {"complete": True, "selection_fingerprint": []}

        self.assertFalse(validation_summary_is_current(workspace, summary))

    def test_stage_four_owns_representative_bin_economics(self) -> None:
        cube = {
            "shift": np.array([[[5.0]], [[12.0]], [[15.0]]]),
            "prob": np.array([[[0.55]], [[0.62]], [[0.65]]]),
            "base": np.array([[0.50], [0.50], [0.50]]),
            "hits": np.array([[[55]], [[62]], [[65]]]),
            "bin_n": np.array([[100]]),
            "Δs": np.array([-0.03, -0.02, -0.01]),
            "horizons": np.array([7]),
        }

        result = economic_filter_sheet(cube, 0, min_dev=10.0, min_bin_n=50, min_run=2)

        self.assertTrue(result["passed"])
        self.assertEqual(result["best"]["Δ"], -0.01)
        self.assertEqual(result["best"]["dev"], 15.0)

    def test_bh_pass_without_raw_null_pass_is_not_a_discovery(self) -> None:
        self.assertEqual(verdict(True, 0.025, False, null_pass=False), "fdr_only")
        self.assertEqual(verdict(True, 0.005, False, null_pass=True), "discovery")


if __name__ == "__main__":
    unittest.main()
