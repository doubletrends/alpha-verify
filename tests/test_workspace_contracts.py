from __future__ import annotations

import unittest
from unittest.mock import patch
from tempfile import TemporaryDirectory
from pathlib import Path

import numpy as np
import pandas as pd

from alphaverify.infrastructure.artifact_history import feature_from_artifact, market_data_from_artifact
from alphaverify.infrastructure.workspace_plugins import load_workspace_module
from alphaverify.infrastructure.workspace import Workspace
from alphaverify.pipeline.context import RunContext


class WorkspaceContractTests(unittest.TestCase):
    def test_catalog_and_stage_paths_match_the_three_stage_pipeline(self) -> None:
        workspace = Workspace("nasdaq_daily")
        self.assertEqual(len(workspace.catalog.all_nodes()), 58)
        self.assertIn(("vix_level", "vix"),
                      [(node["id"], node["family"]) for node in workspace.catalog.all_nodes()])
        self.assertEqual(workspace.cube_path("vix_level").parts[-3:], ("01_surface", "array", "vix_level.safetensors"))
        self.assertEqual(workspace.shift_cube_path("vix_level").parts[-3:], ("02_shift", "array", "vix_level.safetensors"))
        self.assertEqual(workspace.validation_summary_path.parts[-2:], ("03_validation", "validation.json"))
        self.assertEqual(workspace.selection_summary_path.parts[-2:], ("04_selection", "selection.json"))
        self.assertEqual(workspace.forecast_path.parts[-2:], ("05_forecast", "forecast.json"))

    def test_artifact_history_helpers_preserve_rows_and_reject_missing_prices(self) -> None:
        artifact = {
            "index": np.array(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "open": np.array([10.0, 11.0, 12.0]),
            "high": np.array([11.0, 12.0, 13.0]), "low": np.array([9.0, 10.0, 11.0]),
            "close": np.array([10.0, 11.0, 12.0]), "volume": np.ones(3),
            "feature_values": np.array([1.0, 2.0, 3.0]),
        }
        data = market_data_from_artifact(artifact)
        np.testing.assert_array_equal(feature_from_artifact(artifact, data.index).to_numpy(), [1.0, 2.0, 3.0])
        artifact["close"][1] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            market_data_from_artifact(artifact)

    def test_workspace_loader_fetches_shared_ohlcv_only_once_per_run(self) -> None:
        module = load_workspace_module(Workspace("nasdaq_daily").dir, "data", required=True)
        calls = []
        index = pd.date_range("2024-01-01", periods=2)
        def download(ticker, *_args):
            calls.append(ticker)
            return pd.DataFrame({"Open": [10., 11.], "High": [12., 13.],
                                 "Low": [9., 10.], "Close": [11., 12.],
                                 "Volume": [1., 2.]}, index=index)
        with TemporaryDirectory() as directory, patch.object(module, "download", download):
            loader = module.create_loader(start="2024-01-01", cache_dir=Path(directory),
                                          asset={"ticker": "TEST", "interval": "1d", "provider": "yfinance"})
            first = loader(["ohlcv"])
            first.iloc[0, 0] = 999
            second = loader(["ohlcv", "vix"])
            self.assertEqual(second.iloc[0, 0], 10.)
            self.assertEqual(len(list(Path(directory).glob("*.csv"))), 2)
        self.assertEqual(calls, ["TEST", "^VIX"])

    def test_btc_hourly_workspace_uses_its_local_ohlcv_source(self) -> None:
        workspace = Workspace("btc_hourly")
        module = load_workspace_module(workspace.dir, "data", required=True)
        source_frame = pd.DataFrame({"DATETIME": ["2017-12-31T23:00:00Z", "2018-01-01T00:00:00Z"], "OPEN": [100.0, 101.0], "HIGH": [102.0, 103.0], "LOW": [99.0, 100.0], "CLOSE": [101.0, 102.0], "VOLUME_BTC": [10.0, 11.0]})
        with TemporaryDirectory() as directory, patch("pandas.read_csv", return_value=source_frame):
            loader = module.create_loader(start=workspace.start_date, asset=workspace.asset,
                                          cache_dir=Path(directory))
            data = loader(["ohlcv"])
        self.assertEqual(data.columns.tolist(), ["open", "high", "low", "close", "volume"])

    def test_observed_outcomes_are_persisted_and_reused(self) -> None:
        from tempfile import TemporaryDirectory
        from pathlib import Path
        from alphaverify.infrastructure import artifact_io
        from alphaverify.domain import barrier

        with TemporaryDirectory() as directory:
            root = Path(directory) / "workspaces"
            declaration = root / "example" / "universe.json"
            artifact_io.write_json(declaration, {
                "meta": {"asset": {"ticker": "TEST", "interval": "1d"}, "start_date": "2024-01-01",
                         "min_obs": 100, "n_bins": 10,
                         "barriers": {"min": -.02, "max": .02, "step": .02},
                         "horizons": {"min": 1, "max": 3}, "evaluate": {"min_dev": .10, "min_bin_n": 50, "min_run": 2}},
                "families": {},
            })
            ws = Workspace("example", root)
            close = np.linspace(100., 110., 40)
            data = pd.DataFrame({"open": close, "high": close * 1.01, "low": close * .99,
                                 "close": close, "volume": np.ones(40)},
                                index=pd.date_range("2024-01-01", periods=40))
            context = RunContext(ws)
            first = context.observed_outcomes(data)
            cache_files = list((ws.dir / "00_cache").glob("*.safetensors"))
            self.assertEqual(len(cache_files), 1)
            self.assertEqual(first["touch_mask"].shape, (3, 40, 3))
            with patch.object(barrier, "forward_extremes_upto", side_effect=AssertionError("rebuilt")):
                second = RunContext(ws).observed_outcomes(data)
            np.testing.assert_array_equal(second["touch_mask"], first["touch_mask"])


def test_market_history_key_depends_only_on_ohlcv_values_in_canonical_order():
    from alphaverify.infrastructure.artifact_history import market_history_key

    index = pd.date_range("2024-01-01", periods=3, freq="h")
    canonical = pd.DataFrame({"open": [10., 11., 12.], "high": [12., 13., 14.], "low": [9., 10., 11.],
                              "close": [11., 12., 13.], "volume": [1., 2., 3.]}, index=index)
    reordered = canonical[["open", "high", "close", "low", "volume"]]
    with_auxiliary = canonical.assign(vix=[20., 21., np.nan])
    assert market_history_key(canonical) == market_history_key(reordered) == market_history_key(with_auxiliary)
    changed = canonical.copy()
    changed.loc[index[1], "low"] = 9.5
    assert market_history_key(changed) != market_history_key(canonical)
