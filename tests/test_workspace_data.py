"""Prepared-data boundary and workspace policies, with all downloads mocked."""

from pathlib import Path
import shutil
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from alphaverify.infrastructure import artifact_io
from alphaverify.infrastructure.market_data import WorkspaceData, validate_market_data
from alphaverify.infrastructure.workspace import Workspace
from alphaverify.infrastructure.workspace_plugins import load_workspace_module
from alphaverify.pipeline.context import RunContext
from alphaverify.pipeline.step_01_surface import _build_cube


def bars():
    return pd.DataFrame({"open": [10., 11., 12.], "high": [12., 13., 14.],
                         "low": [9., 10., 11.], "close": [11., 12., 13.],
                         "volume": [1., 2., 3.]},
                        index=pd.date_range("2024-01-01", periods=3))


@pytest.mark.parametrize("case", ["empty", "duplicate", "unsorted", "nat", "timezone",
                                 "missing_column", "duplicate_column", "nonnumeric", "missing_price",
                                 "infinite_price", "negative_price", "negative_volume", "bad_high",
                                 "bad_low", "infinite_auxiliary"])
def test_core_rejects_invalid_input_without_repair(case):
    data = bars()
    if case == "empty": data = data.iloc[:0]
    elif case == "duplicate": data.index = [data.index[0]] * len(data)
    elif case == "unsorted": data = data.iloc[::-1]
    elif case == "nat": data.index = pd.DatetimeIndex([pd.NaT, *data.index[1:]])
    elif case == "timezone": data.index = data.index.tz_localize("UTC")
    elif case == "missing_column": data = data.drop(columns="open")
    elif case == "duplicate_column": data = pd.concat([data, data[["close"]]], axis=1)
    elif case == "nonnumeric": data["close"] = ["bad"] * len(data)
    else:
        column, value = {"missing_price": ("close", np.nan), "infinite_price": ("close", np.inf),
                         "negative_price": ("close", -1), "negative_volume": ("volume", -1),
                         "bad_high": ("high", 1), "bad_low": ("low", 100),
                         "infinite_auxiliary": ("vix", np.inf)}[case]
        data.loc[data.index[0], column] = value
    before = data.copy(deep=True)
    with pytest.raises(ValueError):
        validate_market_data(data)
    pd.testing.assert_frame_equal(data, before)


def test_core_keeps_missing_auxiliary_values():
    data = bars()
    data["vix"] = [np.nan, 20., np.nan]
    before = data.copy()
    validate_market_data(data)
    pd.testing.assert_frame_equal(data, before)


def test_missing_workspace_loader_fails_only_when_data_is_requested(tmp_path):
    artifact_io.write_json(tmp_path / "empty" / "universe.json", {
        "meta": {"asset": {"ticker": "TEST"}, "start_date": "2024-01-01"}, "families": {},
    })
    context = RunContext(Workspace("empty", tmp_path))
    with pytest.raises(ValueError, match="must provide data.py"):
        context.load_data(["ohlcv"])


def test_core_calls_workspace_once_and_returns_isolated_panels(monkeypatch):
    import alphaverify.infrastructure.market_data as boundary
    panel = bars()
    loader = Mock(return_value=panel)
    factory = Mock(return_value=loader)
    monkeypatch.setattr(boundary, "load_workspace_module", Mock(return_value=Mock(create_loader=factory)))
    source = WorkspaceData(Workspace("nasdaq_daily"))
    first = source.fetch(["ohlcv"])
    first.iloc[0, 0] = 100
    assert source.fetch(["ohlcv"]).iloc[0, 0] == 10
    loader.assert_called_once_with(["ohlcv"])
    with pytest.raises(ValueError, match="distinct"):
        source.fetch(["ohlcv", "ohlcv"])


@pytest.mark.parametrize("workspace_name", ["nasdaq_daily", "btc_daily"])
def test_daily_policy_deduplicates_drops_missing_bars_and_never_backfills(tmp_path, monkeypatch, workspace_name):
    ws = Workspace(workspace_name)
    module = load_workspace_module(ws.dir, "data", required=True)
    raw = bars().rename(columns=str.title)
    # Last duplicate wins; missing volume causes the entire last bar to be removed.
    duplicate = raw.iloc[[0]].copy()
    duplicate["Volume"] = 9.
    raw.loc[raw.index[-1], "Volume"] = np.nan
    raw = pd.concat([raw, duplicate])
    cross = pd.DataFrame({"Close": [20.]}, index=pd.DatetimeIndex(["2024-01-02"]))
    download = Mock(side_effect=[raw, cross])
    monkeypatch.setattr(module, "download", download)
    loader = module.create_loader(start="2024-01-01", asset=ws.asset, cache_dir=tmp_path)
    panel = loader(["ohlcv", "vix"])
    validate_market_data(panel)
    assert len(panel) == 2
    assert panel.iloc[0]["volume"] == 9
    assert np.isnan(panel.iloc[0]["vix"])
    assert panel.iloc[1]["vix"] == 20
    assert len(list(tmp_path.glob("*.csv"))) == 2
    assert panel.attrs["provenance"]["cleaning_version"] == module.CLEANING_VERSION
    with pytest.raises(ValueError, match="source"):
        loader(["ohlcv", "unknown"])


def test_btc_daily_coinmetrics_preserves_missing_observations(tmp_path, monkeypatch):
    ws = Workspace("btc_daily")
    module = load_workspace_module(ws.dir, "data", required=True)
    monkeypatch.setattr(module, "download", Mock(return_value=bars().rename(columns=str.title)))
    raw = pd.DataFrame({"time": ["2024-01-02T00:00:00Z"],
                        **{key: ["2.5"] for key in module.METRICS}})
    raw["HashRate"] = None
    monkeypatch.setattr(module, "_coinmetrics", Mock(return_value=raw))
    loader = module.create_loader(start="2024-01-01", asset=ws.asset, cache_dir=tmp_path)
    panel = loader(["ohlcv", "coinmetrics"])
    validate_market_data(panel)
    assert np.isnan(panel.iloc[0]["mvrv"])
    assert panel.iloc[-1]["mvrv"] == 2.5
    assert panel["hash_rate"].isna().all()


def test_hourly_policy_converts_utc_and_keeps_last_duplicate(tmp_path, monkeypatch):
    ws = Workspace("btc_hourly")
    module = load_workspace_module(ws.dir, "data", required=True)
    raw = bars().rename(columns=str.upper).rename(columns={"VOLUME": "VOLUME_BTC"}).reset_index(drop=True)
    raw["DATETIME"] = ["2024-01-01T02:00:00+02:00", "2024-01-01T00:00:00Z", "2024-01-01T02:00:00Z"]
    monkeypatch.setattr(pd, "read_csv", Mock(return_value=raw))
    loader = module.create_loader(start="2024-01-01", asset=ws.asset, cache_dir=tmp_path)
    panel = loader(["ohlcv"])
    validate_market_data(panel)
    assert panel.index.tolist() == [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-01T02:00:00")]
    assert panel.iloc[0]["open"] == 11.
    assert panel.attrs["provenance"]["sources"]["ohlcv"]["raw_rows"] == 3


def test_measurement_retains_workspace_provenance(tmp_path, monkeypatch):
    node = {"id": "baseline", "family": "_base", "feature": "constant", "params": {}, "data": ["ohlcv"]}
    artifact_io.write_json(tmp_path / "example" / "universe.json", {
        "meta": {"asset": {"ticker": "TEST"}, "start_date": "2024-01-01", "min_obs": 1,
                 "barriers": {"min": -.02, "max": .02, "step": .02},
                 "horizons": {"min": 1, "max": 1}}, "families": {"_base": [node]},
    })
    ws = Workspace("example", tmp_path)
    context = RunContext(ws)
    data = bars()
    data.attrs["provenance"] = {"cleaning_version": "fixture-v1", "sources": {"ohlcv": {"sha256": "fixture"}}}
    monkeypatch.setattr(context.data, "fetch", Mock(return_value=data))
    _build_cube(context, node)
    assert artifact_io.load_surface(ws.cube_path("baseline"))["meta"]["data_provenance"] == data.attrs["provenance"]


def test_workspace_data_runs_through_all_four_stages_offline(tmp_path, monkeypatch):
    """Exercise real module loading, preparation, snapshots, artifacts, and downstream stages."""
    import yfinance
    from alphaverify.pipeline import step_01_surface, step_02_shift, step_03_validation, step_04_selection

    originals = Path(__file__).parents[1] / "workspaces"
    local = tmp_path / "workspaces"
    (local / "example").mkdir(parents=True)
    (local / "_shared").mkdir()
    shutil.copyfile(originals / "nasdaq_daily" / "data.py", local / "example" / "data.py")
    for filename in ("yahoo.py", "snapshots.py"):
        shutil.copyfile(originals / "_shared" / filename, local / "_shared" / filename)
    nodes = [
        {"id": "baseline", "family": "test", "feature": "constant", "params": {}, "data": ["ohlcv"]},
        {"id": "weekday", "family": "test", "feature": "day_of_week", "params": {}, "data": ["ohlcv"]},
    ]
    artifact_io.write_json(local / "example" / "universe.json", {
        "meta": {"asset": {"ticker": "TEST", "provider": "yfinance", "interval": "1d"},
                 "start_date": "2024-01-01", "min_obs": 30, "n_bins": 2,
                 "barriers": {"min": -.02, "max": .02, "step": .02},
                 "horizons": {"min": 1, "max": 2}}, "families": {"test": nodes},
    })
    rng = np.random.default_rng(42)
    close = 100 * np.exp(np.cumsum(rng.normal(0, .01, 120)))
    raw = pd.DataFrame({"Open": close, "High": close * 1.01, "Low": close * .99,
                        "Close": close, "Volume": np.ones(120)},
                       index=pd.date_range("2024-01-01", periods=120))
    download = Mock(return_value=raw)
    monkeypatch.setattr(yfinance, "download", download)
    monkeypatch.setattr(yfinance, "set_tz_cache_location", Mock())
    monkeypatch.setattr(step_03_validation, "N_NULL_REPLICATES", 16)
    ws = Workspace("example", local)
    step_01_surface.cmd_surface(ws)
    download.assert_called_once()
    assert len(list((ws.dir / "00_data").glob("*.csv"))) == 1
    surface = artifact_io.load_surface(ws.cube_path("weekday"))
    assert surface["meta"]["data_provenance"]["cleaning_version"] == "nasdaq-daily-v1"
    assert surface["meta"]["data_provenance"]["sources"]["ohlcv"]["raw_rows"] == 120
    download.side_effect = AssertionError("downstream stages must use persisted history")
    step_02_shift.cmd_shift(ws)
    step_03_validation.cmd_validation(ws)
    step_04_selection.cmd_selection(ws)
    assert step_03_validation.validation_summary_is_current(ws, ws.read_json(ws.validation_summary_path))
    assert step_04_selection.selection_summary_is_current(ws, ws.read_json(ws.selection_summary_path))
