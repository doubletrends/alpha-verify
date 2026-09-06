from __future__ import annotations

from datetime import date
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from barrierlab.infrastructure.artifacts import (
    feature_from_artifact,
    market_data_from_artifact,
)
from barrierlab.infrastructure.workspace import Workspace
from barrierlab.pipeline.context import RunContext
from barrierlab.pipeline.step_06_composition import composition_targets


class WorkspaceContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workspace = Workspace("nasdaq_daily")

    def test_catalog_is_loaded_once_with_expected_nodes(self) -> None:
        self.assertEqual(len(self.workspace.catalog.all_nodes()), 58)
        self.assertEqual(self.workspace.catalog.find("vix_level")["family"], "vix")
        self.assertNotIn("summary", self.workspace.catalog.meta)
        self.assertEqual(self.workspace.null_alpha, 0.01)
        self.assertEqual(self.workspace.composition_delta, -0.07)
        self.assertEqual(self.workspace.composition_horizon, 30)
        self.assertEqual(self.workspace.composition_folds, 5)
        self.assertEqual(self.workspace.report_as_of, date.today().isoformat())
        self.assertEqual(Workspace("btc_daily").report_as_of, date.today().isoformat())

    def test_composition_targets_cover_the_real_grid(self) -> None:
        targets = composition_targets(self.workspace)
        nonzero_deltas = self.workspace.deltas[self.workspace.deltas != 0]
        self.assertEqual(len(targets), len(nonzero_deltas) * len(self.workspace.horizons))
        self.assertEqual(targets[0], (float(nonzero_deltas[0]), 1))
        self.assertEqual(targets[-1], (float(nonzero_deltas[-1]), 30))
        self.assertTrue(all(delta != 0 for delta, _ in targets))
        self.assertEqual({delta for delta, _ in targets}, set(nonzero_deltas))
        self.assertEqual({horizon for _, horizon in targets}, set(self.workspace.horizons))

    def test_artifact_paths_are_stage_scoped(self) -> None:
        row = {"rank": 7, "family": "vix", "node": "vix_level", "bin_number": 3}
        self.assertEqual(
            self.workspace.cube_path("vix", "vix_level").as_posix().split("/")[-3:],
            ["01_surface", "vix", "vix_level.npz"],
        )
        self.assertEqual(
            self.workspace.surface_path("vix", "vix_level").as_posix().split("/")[-3:],
            ["01_surface", "vix", "vix_level.xlsx"],
        )
        self.assertEqual(
            self.workspace.shift_cube_path("vix", "vix_level").as_posix().split("/")[-3:],
            ["02_shift", "vix", "vix_level.npz"],
        )
        self.assertEqual(
            self.workspace.shift_surface_path("vix", "vix_level").as_posix().split("/")[-3:],
            ["02_shift", "vix", "vix_level.xlsx"],
        )
        self.assertEqual(
            self.workspace.selection_array_path(row).as_posix().split("/")[-3:],
            ["03_selection", "vix", "rank_007__vix_level__bin_03.npz"],
        )
        self.assertEqual(
            self.workspace.selection_surface_path(row).as_posix().split("/")[-3:],
            ["03_selection", "vix", "rank_007__vix_level__bin_03.xlsx"],
        )
        self.assertEqual(
            self.workspace.validation_array_path(row).as_posix().split("/")[-3:],
            ["04_validation", "vix", "rank_007__vix_level__bin_03.npz"],
        )
        self.assertEqual(
            self.workspace.validation_surface_path(row).as_posix().split("/")[-3:],
            ["04_validation", "vix", "rank_007__vix_level__bin_03.xlsx"],
        )
        self.assertEqual(
            self.workspace.redundancy_path.as_posix().split("/")[-2:],
            ["05_redundancy", "redundancy.json"],
        )
        self.assertEqual(
            self.workspace.redundancy_array_path.as_posix().split("/")[-2:],
            ["05_redundancy", "redundancy.npz"],
        )
        self.assertEqual(
            self.workspace.redundancy_workbook_path.as_posix().split("/")[-2:],
            ["05_redundancy", "redundancy.xlsx"],
        )
        self.assertEqual(
            self.workspace.validation_summary_path.as_posix().split("/")[-2:],
            ["04_validation", "validation.json"],
        )
        self.assertEqual(
            self.workspace.composition_probability_workbook_path.as_posix().split("/")[
                -2:
            ],
            ["06_composition", "probability.xlsx"],
        )
        self.assertEqual(
            self.workspace.composition_shift_workbook_path.as_posix().split("/")[-2:],
            ["06_composition", "shift.xlsx"],
        )
        self.assertEqual(
            self.workspace.composition_array_path.as_posix().split("/")[-2:],
            ["06_composition", "composition.npz"],
        )
        self.assertEqual(
            self.workspace.composition_summary_path.as_posix().split("/")[-2:],
            ["06_composition", "composition.json"],
        )
        self.assertEqual(
            self.workspace.report_dir.as_posix().split("/")[-1],
            "07_report",
        )

    def test_artifact_history_helpers_align_feature_to_valid_prices(self) -> None:
        artifact = {
            "index": np.array(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "high": np.array([11.0, 12.0, 13.0]),
            "low": np.array([9.0, 10.0, 11.0]),
            "close": np.array([10.0, np.nan, 12.0]),
            "feature_values": np.array([1.0, 2.0, 3.0]),
        }
        data = market_data_from_artifact(artifact)
        feature = feature_from_artifact(artifact, data.index)
        self.assertEqual(len(data), 2)
        np.testing.assert_array_equal(feature.to_numpy(), np.array([1.0, 3.0]))

    def test_artifact_history_helper_rejects_incomplete_history(self) -> None:
        with self.assertRaisesRegex(ValueError, "ordered history"):
            market_data_from_artifact({"index": np.array(["2024-01-01"])})

    def test_workspace_plugin_registers_into_its_run_context(self) -> None:
        context = RunContext(Workspace("btc_daily"))
        data = pd.DataFrame({"mvrv": [1.1]}, index=pd.to_datetime(["2024-01-01"]))
        result = context.features.compute(data, "mvrv", {})
        self.assertEqual(float(result.iloc[0]), 1.1)

    def test_btc_hourly_workspace_uses_its_local_raw_ohlcv_source(self) -> None:
        workspace = Workspace("btc_hourly")
        context = RunContext(workspace)
        source_frame = pd.DataFrame(
            {
                "DATETIME": ["2017-12-31T23:00:00Z", "2018-01-01T00:00:00Z"],
                "OPEN": [100.0, 101.0],
                "HIGH": [102.0, 103.0],
                "LOW": [99.0, 100.0],
                "CLOSE": [101.0, 102.0],
                "VOLUME_BTC": [10.0, 11.0],
            }
        )

        with patch("pandas.read_csv", return_value=source_frame) as read_csv:
            data = context.sources.fetch(["ohlcv"], workspace.start_date, workspace.asset)

        read_csv.assert_called_once()
        self.assertEqual(data.index.tolist(), [pd.Timestamp("2018-01-01 00:00:00")])
        self.assertEqual(data.columns.tolist(), ["open", "high", "low", "close", "volume"])
        self.assertEqual(workspace.horizon_unit, "h")
        self.assertEqual((workspace.horizons[0], workspace.horizons[-1]), (1, 48))


if __name__ == "__main__":
    unittest.main()
