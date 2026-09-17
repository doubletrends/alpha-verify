"""Prepared-data boundary and workspace policies, with all downloads mocked."""

from pathlib import Path
import shutil
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from alphaverify.infrastructure import artifact_io
from alphaverify.infrastructure.artifact_history import market_history_key
from alphaverify.infrastructure.market_data import WorkspaceData, validate_market_data
from alphaverify.infrastructure.workspace import Workspace
from alphaverify.infrastructure.workspace_plugins import load_workspace_module
from alphaverify.pipeline.step_01_surface import cmd_surface


def bars():
    return pd.DataFrame({"open": [10., 11., 12.], "high": [12., 13., 14.],
                         "low": [9., 10., 11.], "close": [11., 12., 13.],
                         "volume": [1., 2., 3.]},
                        index=pd.date_range("2024-01-01", periods=3))


def with_value(column, value):
    def mutate(data):
        data.loc[data.index[0], column] = value
        return data
    return mutate


def with_index(index):
    def mutate(data):
        data.index = index(data.index)
        return data
    return mutate


# One case per rejection condition in validate_market_data.
INVALID_INPUTS = {
    "empty": lambda data: data.iloc[:0],
    "duplicate_timestamp": with_index(lambda index: [index[0]] * len(index)),
    "unsorted": lambda data: data.iloc[::-1],
    "missing_timestamp": with_index(lambda index: pd.DatetimeIndex([pd.NaT, *index[1:]])),
    "timezone": with_index(lambda index: index.tz_localize("UTC")),
    "missing_column": lambda data: data.drop(columns="open"),
    "duplicate_column": lambda data: pd.concat([data, data[["close"]]], axis=1),
    "nonnumeric": lambda data: data.assign(close=["bad"] * len(data)),
    "missing_price": with_value("close", np.nan),
    "negative_price": with_value("close", -1),
    "negative_volume": with_value("volume", -1),
    "high_below_close": with_value("high", 1),
    "low_above_close": with_value("low", 100),
    "infinite_auxiliary": with_value("vix", np.inf),
}


@pytest.mark.parametrize("mutate", INVALID_INPUTS.values(), ids=INVALID_INPUTS.keys())
def test_core_rejects_invalid_input_without_repair(mutate):
    data = mutate(bars())
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


def test_btc_daily_excludes_the_forming_utc_day(tmp_path, monkeypatch):
    ws = Workspace("btc_daily")
    module = load_workspace_module(ws.dir, "data", required=True)
    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    raw = bars().rename(columns=str.title)
    raw.index = pd.date_range(end=today, periods=len(raw))
    raw.loc[today, "Close"] = raw.loc[today, "High"] + 1  # A live bar can outrun its high.
    monkeypatch.setattr(module, "download", Mock(return_value=raw))
    loader = module.create_loader(start="2024-01-01", asset=ws.asset, cache_dir=tmp_path)
    panel = loader(["ohlcv"])
    validate_market_data(panel)
    assert panel.index[-1] == today - pd.Timedelta(days=1)


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


def test_hourly_policy_drops_invalid_ohlc_bars_and_the_forming_hour(tmp_path, monkeypatch):
    ws = Workspace("btc_hourly")
    module = load_workspace_module(ws.dir, "data", required=True)
    current_hour = pd.Timestamp.now(tz="UTC").tz_localize(None).floor("h")
    hours = [current_hour - pd.Timedelta(hours=offset) for offset in (3, 2, 1, 0)]
    raw = pd.DataFrame({
        "DATETIME": [f"{hour.isoformat()}Z" for hour in hours],
        "OPEN": [10., 11., 12., 13.], "HIGH": [12., 13., 14., 15.],
        "LOW": [9., 11.5, 11., 12.], "CLOSE": [11., 12., 13., 14.], "VOLUME_BTC": [1., 2., 3., 4.],
    })  # The second bar's low sits above its open; the last bar's hour has not ended.
    monkeypatch.setattr(pd, "read_csv", Mock(return_value=raw))
    loader = module.create_loader(start="2024-01-01", asset=ws.asset, cache_dir=tmp_path)
    panel = loader(["ohlcv"])
    validate_market_data(panel)
    assert panel.index.tolist() == [hours[0], hours[2]]
    assert panel.attrs["provenance"]["sources"]["ohlcv"]["invalid_ohlc_bars_dropped"] == [str(hours[1])]


def test_measurement_retains_workspace_provenance(tmp_path, monkeypatch):
    node = {"id": "baseline", "family": "_base", "feature": "constant", "params": {}, "data": ["ohlcv"]}
    artifact_io.write_json(tmp_path / "example" / "universe.json", {
        "meta": {"asset": {"ticker": "TEST", "interval": "1d"}, "start_date": "2024-01-01",
                 "min_obs": 1, "n_bins": 10, "barriers": {"min": -.02, "max": .02, "step": .02},
                 "horizons": {"min": 1, "max": 1}, "evaluate": {"min_dev": .10, "min_bin_n": 50, "min_run": 2}},
        "families": {"_base": [node]},
    })
    ws = Workspace("example", tmp_path)
    data = bars()
    data.attrs["provenance"] = {"cleaning_version": "fixture-v1", "sources": {"ohlcv": {"sha256": "fixture"}}}
    monkeypatch.setattr(WorkspaceData, "fetch", Mock(return_value=data))
    cmd_surface(ws)
    assert artifact_io.load_surface(ws.cube_path("baseline"))["meta"]["data_provenance"] == data.attrs["provenance"]


def test_workspace_data_runs_through_all_five_stages_offline(tmp_path, monkeypatch):
    """Exercise real module loading, preparation, snapshots, artifacts, and downstream stages."""
    import yfinance
    from alphaverify.pipeline import (
        step_01_surface, step_02_shift, step_03_validation, step_04_selection, step_05_forecast,
    )

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
                 "horizons": {"min": 1, "max": 2}, "evaluate": {"min_dev": .10, "min_bin_n": 50, "min_run": 2}},
        "families": {"test": nodes},
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
    download.side_effect = AssertionError("downstream stages must use persisted history")
    step_02_shift.cmd_shift(ws)
    step_03_validation.cmd_validation(ws)
    step_04_selection.cmd_selection(ws)
    step_05_forecast.cmd_forecast(ws)
    assert step_03_validation.validation_summary_is_current(ws, artifact_io.read_json(ws.validation_summary_path))
    assert step_04_selection.selection_summary_is_current(ws, artifact_io.read_json(ws.selection_summary_path))
    assert artifact_io.read_json(ws.forecast_path)["complete"]


def test_workspace_loader_fetches_shared_ohlcv_only_once_per_run(tmp_path, monkeypatch):
    # Regression for c5d173c: market data was downloaded again for every source request.
    module = load_workspace_module(Workspace("nasdaq_daily").dir, "data", required=True)
    calls = []

    def download(ticker, *_args):
        calls.append(ticker)
        return bars().rename(columns=str.title)

    monkeypatch.setattr(module, "download", download)
    loader = module.create_loader(start="2024-01-01", cache_dir=tmp_path,
                                  asset={"ticker": "TEST", "interval": "1d", "provider": "yfinance"})
    loader(["ohlcv"])
    loader(["ohlcv", "vix"])
    assert calls == ["TEST", "^VIX"]


def test_market_history_key_depends_only_on_ohlcv_values_in_canonical_order():
    canonical = bars()
    reordered = canonical[["open", "high", "close", "low", "volume"]]
    with_auxiliary = canonical.assign(vix=[20., 21., np.nan])
    assert market_history_key(canonical) == market_history_key(reordered) == market_history_key(with_auxiliary)
    changed = canonical.copy()
    changed.iloc[1, changed.columns.get_loc("low")] = 9.5
    assert market_history_key(changed) != market_history_key(canonical)
