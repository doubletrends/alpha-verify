from __future__ import annotations

import unittest
from pathlib import Path

from barrierlab.domain.features import is_ohlcv_feature
from barrierlab.pipeline.step_04_validation import validation_summary_is_current


class _Workspace:
    selection_path = Path("selection.json")

    def read_json(self, _path: Path) -> dict:
        return {"selected": [{
            "rank": 1, "node": "atr", "bin": 9, "delta": 0.10,
            "horizon": 14, "score": 0.02,
        }]}


class CombinedValidationTests(unittest.TestCase):
    def test_only_core_ohlcv_features_are_recomputed_for_the_null(self) -> None:
        self.assertTrue(is_ohlcv_feature("atr"))
        self.assertFalse(is_ohlcv_feature("days_since_halving"))
        self.assertFalse(is_ohlcv_feature("day_of_week"))

    def test_stale_selection_fingerprint_is_rejected(self) -> None:
        self.assertFalse(validation_summary_is_current(_Workspace(), {"complete": True}))

    def test_matching_simulated_summary_is_current(self) -> None:
        summary = {
            "complete": True,
            "selection_fingerprint": [{
                "rank": 1, "node": "atr", "bin": 9, "delta": 0.10,
                "horizon": 14, "score": 0.02,
            }],
            "method": {"unit": "one full-grid linearly Δ-weighted condition-bin score; raw p < 0.05"},
        }
        self.assertTrue(validation_summary_is_current(_Workspace(), summary))
