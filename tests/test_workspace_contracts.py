from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from barrierlab.infrastructure.artifacts import feature_from_artifact, market_data_from_artifact
from barrierlab.infrastructure.market_data import SourceRegistry
from barrierlab.infrastructure.workspace import Workspace
from barrierlab.pipeline.context import RunContext


class WorkspaceContractTests(unittest.TestCase):
    def test_catalog_and_stage_paths_match_the_three_stage_pipeline(self) -> None:
        workspace = Workspace("nasdaq_daily")
        self.assertEqual(len(workspace.catalog.all_nodes()), 58)
        self.assertEqual(workspace.catalog.find("vix_level")["family"], "vix")
        self.assertEqual(workspace.cube_path("vix_level").parts[-3:], ("01_surface", "array", "vix_level.safetensors"))
        self.assertEqual(workspace.shift_cube_path("vix_level").parts[-3:], ("02_shift", "array", "vix_level.safetensors"))
        self.assertEqual(workspace.validation_summary_path.parts[-2:], ("03_validation", "validation.json"))

    def test_artifact_history_helpers_align_feature_to_valid_prices(self) -> None:
        artifact = {
            "index": np.array(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "high": np.array([11.0, 12.0, 13.0]), "low": np.array([9.0, 10.0, 11.0]),
            "close": np.array([10.0, np.nan, 12.0]), "feature_values": np.array([1.0, 2.0, 3.0]),
        }
        data = market_data_from_artifact(artifact)
        np.testing.assert_array_equal(feature_from_artifact(artifact, data.index).to_numpy(), [1.0, 3.0])

    def test_source_registry_fetches_shared_ohlcv_only_once_per_run(self) -> None:
        registry, calls = SourceRegistry(), []
        index = pd.date_range("2024-01-01", periods=2)
        def source(name, columns):
            def fetch(**_kwargs):
                calls.append(name)
                return pd.DataFrame(columns, index=index)
            return fetch
        registry.register("ohlcv", source("ohlcv", {"close": [1.0, 2.0]}))
        registry.register("vix", source("vix", {"vix": [10.0, 11.0]}))
        asset = {"ticker": "TEST", "interval": "1d"}
        registry.fetch(["ohlcv"], "2024-01-01", asset)
        registry.fetch(["ohlcv", "vix"], "2024-01-01", asset)
        self.assertEqual(calls, ["ohlcv", "vix"])

    def test_btc_hourly_workspace_uses_its_local_ohlcv_source(self) -> None:
        workspace, context = Workspace("btc_hourly"), RunContext(Workspace("btc_hourly"))
        source_frame = pd.DataFrame({"DATETIME": ["2017-12-31T23:00:00Z", "2018-01-01T00:00:00Z"], "OPEN": [100.0, 101.0], "HIGH": [102.0, 103.0], "LOW": [99.0, 100.0], "CLOSE": [101.0, 102.0], "VOLUME_BTC": [10.0, 11.0]})
        with patch("pandas.read_csv", return_value=source_frame):
            data = context.sources.fetch(["ohlcv"], workspace.start_date, workspace.asset)
        self.assertEqual(data.columns.tolist(), ["open", "high", "low", "close", "volume"])
