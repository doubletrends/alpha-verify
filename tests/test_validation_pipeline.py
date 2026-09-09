"""Validation runs from Stage 2, tests every eligible bin, and tracks provenance."""

from copy import deepcopy
from dataclasses import replace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from barrierlab.domain import barrier, shift, validation
from barrierlab.domain.features import is_ohlcv_feature
from barrierlab.infrastructure import artifact_io
from barrierlab.infrastructure.workspace import Workspace
from barrierlab.pipeline import step_03_validation as stage
from barrierlab.pipeline.status import cmd_status
from barrierlab.presentation import validation_plots


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    nodes = [
        {"id": name, "family": "test", "feature": "day_of_week", "params": {}}
        for name in ("a", "b", "constant")
    ]
    path = tmp_path / "workspaces" / "example" / "universe.json"
    artifact_io.write_json(path, {
        "meta": {"asset": {"ticker": "TEST"}, "start_date": "2024-01-01", "n_bins": 2},
        "families": {"test": nodes},
    })
    ws = Workspace("example", tmp_path / "workspaces")
    rng = np.random.default_rng(12)
    close = 100 * np.exp(np.cumsum(rng.normal(0, .02, 160)))
    data = pd.DataFrame({"open": close, "high": close * 1.01, "low": close * .99,
                         "close": close, "volume": np.ones(160)},
                        index=pd.date_range("2024-01-01", periods=160))
    deltas, horizons = np.array([-.02, .02]), np.array([1, 3])
    for node in nodes:
        feature = pd.Series(np.ones(160) if node["id"] == "constant" else np.arange(160) % 2,
                            index=data.index)
        cube = barrier.touch_tensor(data, feature, horizons, deltas, barrier.bin_edges(feature, 2))
        baseline = barrier.touch_tensor(data, pd.Series(1., index=data.index),
                                        horizons, deltas, np.array([]))["prob"][:, 0, :]
        cube = shift.from_cube(cube, baseline)
        cube.update({key: data[key].to_numpy() for key in data})
        cube.update(index=data.index.astype(str).to_numpy(), feature_values=feature.to_numpy())
        artifact_io.save_shift(cube, ws.shift_cube_path(node["id"]), {"bin_labels": ["low", "high"]})
    monkeypatch.setattr(stage, "N_PATHS", 16)
    monkeypatch.setattr(validation_plots, "write_bin_score_null_histograms", Mock(return_value=[]))
    return ws


def test_only_core_features_are_recomputed():
    assert is_ohlcv_feature("atr")
    assert not is_ohlcv_feature("days_since_halving")
    assert not is_ohlcv_feature("day_of_week")


def test_all_bins_are_validated_without_selection(workspace, monkeypatch, capsys):
    simulate = Mock(wraps=validation.simulated_ohlc_tensor)
    monkeypatch.setattr(validation, "simulated_ohlc_tensor", simulate)
    stage.cmd_validation(workspace)
    summary = workspace.read_json(workspace.validation_summary_path)
    assert summary["complete"]
    assert {(row["node"], row["bin"]) for row in summary["tests"]} == {
        ("a", 0), ("a", 1), ("b", 0), ("b", 1),
    }
    assert summary["skipped_bins"][0]["node"] == "constant"
    assert summary["summary"]["skipped"] == 1
    assert simulate.call_count == 1
    for row in summary["tests"]:
        assert "rank" not in row and "best_cell" not in row
        assert len(row["null_scores"]) == 16
        expected = (1 + sum(score >= row["bin_score"] for score in row["null_scores"])) / 17
        assert row["peak_p"] == expected
    assert stage.validation_summary_is_current(workspace, summary)
    cmd_status(workspace)
    assert "4 condition bins" in capsys.readouterr().out


def test_different_market_histories_get_separate_nulls(workspace, monkeypatch):
    path = workspace.shift_cube_path("b")
    cube = artifact_io.load_shift(path)
    for key in ("open", "high", "low", "close"):
        cube[key] *= 2
    artifact_io.save_shift(cube, path, cube["meta"])
    simulate = Mock(wraps=validation.simulated_ohlc_tensor)
    monkeypatch.setattr(validation, "simulated_ohlc_tensor", simulate)
    stage.cmd_validation(workspace)
    assert simulate.call_count == 2


def test_freshness_tracks_artifact_bytes_catalog_and_settings(workspace, monkeypatch):
    stage.cmd_validation(workspace)
    summary = workspace.read_json(workspace.validation_summary_path)
    for key in ("seed", "n_paths", "scoring_version", "measurement_version", "null_version"):
        changed = deepcopy(summary)
        changed["method"][key] = "old"
        assert not stage.validation_summary_is_current(workspace, changed)
    changed = deepcopy(summary)
    changed["artifact"] = "04_validation"
    assert not stage.validation_summary_is_current(workspace, changed)
    original_config = workspace.config
    workspace.config = replace(original_config, n_bins=3)
    assert not stage.validation_summary_is_current(workspace, summary)
    workspace.config = original_config
    workspace.catalog.raw["families"]["test"][0]["params"]["changed"] = True
    assert not stage.validation_summary_is_current(workspace, summary)
    workspace.catalog.raw["families"]["test"][0]["params"].clear()
    path = workspace.shift_cube_path("a")
    cube = artifact_io.load_shift(path)
    cube["prob"][0, 0, 0] += .01
    artifact_io.save_shift(cube, path, cube["meta"])
    assert not stage.validation_summary_is_current(workspace, summary)


def test_missing_artifact_is_incomplete_and_invalidates_prior_result(workspace):
    stage.cmd_validation(workspace)
    previous = workspace.read_json(workspace.validation_summary_path)
    workspace.shift_cube_path("b").unlink()
    assert not stage.validation_summary_is_current(workspace, previous)
    stage.cmd_validation(workspace)
    summary = workspace.read_json(workspace.validation_summary_path)
    assert not summary["complete"]
    assert summary["missing_nodes"] == ["b"]
    assert len(summary["tests"]) == 2


def test_valid_zero_bins_are_tested(workspace):
    for node in ("a", "b"):
        path = workspace.shift_cube_path(node)
        cube = artifact_io.load_shift(path)
        for key in ("open", "high", "low", "close"):
            cube[key][:] = 100.0
        artifact_io.save_shift(cube, path, cube["meta"])
    stage.cmd_validation(workspace)
    summary = workspace.read_json(workspace.validation_summary_path)
    assert len(summary["tests"]) == 4
    assert all(row["bin_score"] == 0 and row["peak_p"] == 1 for row in summary["tests"])


@pytest.mark.parametrize("feature_name", ["roc", "day_of_week"])
def test_identical_history_has_same_score_and_validity_as_observed_or_null(
    workspace, monkeypatch, feature_name,
):
    """Exercise the actual observed/null call sites, including artifact loading.

    Corrupt presentation arrays to catch any return to stored-probability
    scoring. Put the observed history twice inside a mixed null batch so the
    assertion also protects against batch-position and own-baseline mistakes.
    """
    node = workspace.catalog.raw["families"]["test"][0]
    workspace.catalog.raw["families"]["test"] = [node]
    node.update(feature=feature_name, params={"period": 5} if feature_name == "roc" else {})
    path = workspace.shift_cube_path("a")
    cube = artifact_io.load_shift(path)
    cube["prob"][:] = np.nan
    cube["base"][:] = np.nan
    cube["shift"][:] = 999
    cube["bin_n"][:] = 0
    if feature_name == "roc":
        cube["feature_values"] = np.full(len(cube["feature_values"]), np.nan)
        cube["edges"] = np.array([-999., 999.])
    else:
        cube["feature_values"] = cube["feature_values"].astype(float)
        cube["feature_values"][:40] = np.nan
        cube["edges"] = np.array([0., 1.])  # Ties plus an unsupported final bin.
    artifact_io.save_shift(cube, path, {"bin_labels": ["STALE LABEL"] * 3})

    def null_with_observed_history(data, n_paths, seed):
        actual = barrier.ohlcv_tensor(data)
        batch = actual.expand(n_paths, -1, -1).clone()
        # Different intrabar ranges in other histories must not change the
        # baseline or score of the identical first and last histories.
        batch[1:-1, :, 1] *= 1.2
        batch[1:-1, :, 2] *= .8
        return batch

    monkeypatch.setattr(validation, "simulated_ohlc_tensor", null_with_observed_history)
    calls, batch_calls = [], []
    score_histories = validation.score_histories
    score_histories_many = validation.score_histories_many

    def capture(paths, **policy):
        result = score_histories(paths, **policy)
        calls.append((result, policy))
        return result

    monkeypatch.setattr(validation, "score_histories", capture)
    def capture_many(paths, policies, **kwargs):
        result = score_histories_many(paths, policies, **kwargs)
        batch_calls.extend(zip(result, policies))
        return result

    monkeypatch.setattr(validation, "score_histories_many", capture_many)
    stage.cmd_validation(workspace)
    assert len(calls) == 1 and len(batch_calls) == 1
    observed, null = calls[0][0], batch_calls[0][0]
    for position in (0, -1):
        np.testing.assert_allclose(observed["scores"][0], null["scores"][position], rtol=0, atol=1e-10)
        np.testing.assert_array_equal(observed["valid"][0], null["valid"][position])
        np.testing.assert_array_equal(observed["edges"][0], null["edges"][position])
    assert observed["valid"].any()
    if feature_name == "day_of_week":
        assert observed["valid"].tolist() == [[True, True, False]]
    summary = workspace.read_json(workspace.validation_summary_path)
    assert summary["tests"]
    assert all(row["bin_label"] != "STALE LABEL" for row in summary["tests"])
    for row in summary["tests"]:
        assert row["bin_score"] == observed["scores"][0, row["bin"]]


def test_plot_uses_node_bin_identity_without_rank(workspace, monkeypatch):
    # Restore the actual renderer and exercise it with a complete stage result.
    import importlib
    renderer = importlib.reload(validation_plots).write_bin_score_null_histograms
    monkeypatch.setattr(validation_plots, "write_bin_score_null_histograms", renderer)
    stage.cmd_validation(workspace)
    plots = list((workspace.validation_summary_path.parent / "plot").glob("*.png"))
    assert len(plots) == 4
    assert {p.name for p in plots} == {
        f"null_histogram__{node}__bin_{b:02d}.png" for node in ("a", "b") for b in (1, 2)
    }
    assert all(p.stat().st_size > 1000 for p in plots)
    workspace.shift_cube_path("b").unlink()
    stage.cmd_validation(workspace)
    assert len(list((workspace.validation_summary_path.parent / "plot").glob("*.png"))) == 2
