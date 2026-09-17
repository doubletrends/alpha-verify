"""Validation runs from Stage 2, tests every eligible bin, and tracks provenance."""

from dataclasses import replace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from alphaverify.domain import barrier, shift, validation
from alphaverify.domain.features import FeatureRegistry
from alphaverify.infrastructure import artifact_io
from alphaverify.infrastructure.artifact_history import market_data_from_artifact
from alphaverify.infrastructure.workspace import Workspace
from alphaverify.pipeline import step_03_validation as stage
from alphaverify.presentation import bin_figures


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    nodes = [
        {"id": name, "family": "test", "feature": "day_of_week", "params": {}}
        for name in ("a", "b", "constant")
    ]
    path = tmp_path / "workspaces" / "example" / "universe.json"
    artifact_io.write_json(path, {
        "meta": {"asset": {"ticker": "TEST", "interval": "1d"}, "start_date": "2024-01-01",
                 "min_obs": 100, "n_bins": 2, "barriers": {"min": -.20, "max": .20, "step": .01},
                 "horizons": {"min": 1, "max": 30}, "evaluate": {"min_dev": .10, "min_bin_n": 50, "min_run": 2}},
        "families": {"test": nodes},
    })
    ws = Workspace("example", tmp_path / "workspaces")
    rng = np.random.default_rng(12)
    close = 100 * np.exp(np.cumsum(rng.normal(0, .02, 160)))
    data = pd.DataFrame({"open": close, "high": close * 1.01, "low": close * .99,
                         "close": close, "volume": np.ones(160)},
                        index=pd.date_range("2024-01-01", periods=160))
    deltas, horizons = np.array([-.02, .02]), np.array([1, 3])
    history = {key: data[key].to_numpy() for key in data}
    history["index"] = data.index.astype(str).to_numpy()
    baseline = barrier.touch_tensor(data, pd.Series(1., index=data.index),
                                    horizons, deltas, np.array([]))
    artifact_io.save_surface({**baseline, **history}, ws.baseline_cube, {"bin_labels": ["all"]})
    for node in nodes:
        feature = pd.Series(np.ones(160) if node["id"] == "constant" else np.arange(160) % 2,
                            index=data.index)
        cube = barrier.touch_tensor(data, feature, horizons, deltas, barrier.bin_edges(feature, 2))
        cube.update(history, feature_values=feature.to_numpy())
        artifact_io.save_surface(cube, ws.cube_path(node["id"]), {"bin_labels": ["low", "high"]})
        artifact_io.save_shift(
            shift.from_cube(cube, baseline["conditional_probability"][:, 0, :]),
            ws.shift_cube_path(node["id"]), {"bin_labels": ["low", "high"]},
        )
    monkeypatch.setattr(stage, "N_NULL_REPLICATES", 16)
    monkeypatch.setattr(bin_figures, "write_bin_figures", Mock(return_value=[]))
    return ws


def test_all_bins_are_validated_without_selection(workspace):
    stage.cmd_validation(workspace)
    summary = artifact_io.read_json(workspace.validation_summary_path)
    assert {(row["node"], row["bin"]) for row in summary["tests"]} == {
        ("a", 0), ("a", 1), ("b", 0), ("b", 1),
    }
    assert [row["node"] for row in summary["skipped_bins"]] == ["constant"]


def test_monte_carlo_p_value_counts_null_scores_at_least_as_large(workspace):
    stage.cmd_validation(workspace)
    row = artifact_io.read_json(workspace.validation_summary_path)["tests"][0]
    expected = (1 + sum(score >= row["bin_score"] for score in row["null_scores"])) / (len(row["null_scores"]) + 1)
    assert row["monte_carlo_p_value"] == expected


def test_different_market_histories_get_separate_nulls(workspace, monkeypatch):
    path = workspace.cube_path("b")
    cube = artifact_io.load_surface(path)
    for key in ("open", "high", "low", "close"):
        cube[key] *= 2
    artifact_io.save_surface(cube, path, cube["meta"])
    simulate = Mock(wraps=validation.simulated_ohlc_tensor)
    monkeypatch.setattr(validation, "simulated_ohlc_tensor", simulate)
    stage.cmd_validation(workspace)
    assert simulate.call_count == 2


def change_method(workspace, summary):
    summary["method"]["seed"] = "old"


def change_config(workspace, summary):
    workspace.config = replace(workspace.config, n_bins=3)


def change_catalog(workspace, summary):
    workspace.catalog.raw["families"]["test"][0]["params"]["changed"] = True


def change_artifact_bytes(workspace, summary):
    path = workspace.cube_path("a")
    cube = artifact_io.load_surface(path)
    cube["conditional_probability"][0, 0, 0] += .01
    artifact_io.save_surface(cube, path, cube["meta"])


@pytest.mark.parametrize("change", [change_method, change_config, change_catalog, change_artifact_bytes])
def test_validation_goes_stale_when_an_input_changes(workspace, change):
    stage.cmd_validation(workspace)
    summary = artifact_io.read_json(workspace.validation_summary_path)
    change(workspace, summary)
    assert not stage.validation_summary_is_current(workspace, summary)


def test_missing_artifact_is_incomplete_and_invalidates_prior_result(workspace):
    stage.cmd_validation(workspace)
    previous = artifact_io.read_json(workspace.validation_summary_path)
    workspace.shift_cube_path("b").unlink()
    assert not stage.validation_summary_is_current(workspace, previous)
    stage.cmd_validation(workspace)
    summary = artifact_io.read_json(workspace.validation_summary_path)
    assert not summary["complete"]
    assert summary["missing_nodes"] == ["b"]
    assert len(summary["tests"]) == 2


def test_valid_zero_bins_are_tested(workspace):
    for node in ("a", "b"):
        path = workspace.cube_path(node)
        cube = artifact_io.load_surface(path)
        for key in ("open", "high", "low", "close"):
            cube[key][:] = 100.0
        artifact_io.save_surface(cube, path, cube["meta"])
    stage.cmd_validation(workspace)
    summary = artifact_io.read_json(workspace.validation_summary_path)
    assert len(summary["tests"]) == 4
    assert all(
        row["bin_score"] == 0 and row["monte_carlo_p_value"] == 1
        for row in summary["tests"]
    )


@pytest.mark.parametrize("feature_name", ["roc", "day_of_week"])
def test_identical_history_has_same_score_and_validity_as_observed_or_null(
    workspace, monkeypatch, feature_name,
):
    """Exercise the actual observed/null call sites, including artifact loading.

    Corrupt stored probabilities, counts, and shifts to catch any return to
    stored-probability scoring. The observed role reuses the Stage 1 condition,
    which the null recomputes for a core feature. Put the observed history twice
    inside a mixed null batch so the assertion also protects against
    batch-position and own-baseline mistakes.
    """
    node = workspace.catalog.raw["families"]["test"][0]
    workspace.catalog.raw["families"]["test"] = [node]
    node.update(feature=feature_name, params={"period": 5} if feature_name == "roc" else {})
    path = workspace.cube_path("a")
    cube = artifact_io.load_surface(path)
    cube["conditional_probability"][:] = np.nan
    cube["bin_observation_counts"][:] = 0
    if feature_name == "roc":
        values = FeatureRegistry().compute(market_data_from_artifact(cube), "roc", node["params"])
        cube["feature_values"] = values.to_numpy()
        cube["bin_edges"] = barrier.bin_edges(values, 2)
    else:
        cube["feature_values"] = cube["feature_values"].astype(float)
        cube["feature_values"][:40] = np.nan
        cube["bin_edges"] = np.array([0., 1.])  # Ties plus an unsupported final bin.
    cube["bin_assignments"] = np.searchsorted(
        cube["bin_edges"], cube["feature_values"], side="left"
    ).astype(np.uint8)
    artifact_io.save_surface(cube, path, {"bin_labels": ["STALE LABEL"] * 3})
    baseline = artifact_io.load_surface(workspace.baseline_cube)
    baseline["conditional_probability"][:] = np.nan
    artifact_io.save_surface(baseline, workspace.baseline_cube, baseline["meta"])
    shifted = artifact_io.load_shift(workspace.shift_cube_path("a"))
    artifact_io.save_shift(
        np.full_like(shifted["probability_shift"], 999), workspace.shift_cube_path("a"), {},
    )

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
        np.testing.assert_allclose(
            observed.bin_score[0], null.bin_score[position], rtol=0, atol=1e-10
        )
        np.testing.assert_array_equal(
            observed.score_supported[0], null.score_supported[position]
        )
        np.testing.assert_array_equal(observed.bin_edges[0], null.bin_edges[position])
    assert observed.score_supported.any()
    if feature_name == "day_of_week":
        assert observed.score_supported.tolist() == [[True, True, False]]
    summary = artifact_io.read_json(workspace.validation_summary_path)
    assert summary["tests"]
    assert all(row["bin_label"] != "STALE LABEL" for row in summary["tests"])
    for row in summary["tests"]:
        assert row["bin_score"] == observed.bin_score[0, row["bin"]]


def test_shift_without_the_current_units_is_rejected(tmp_path):
    path = tmp_path / "shift.safetensors"
    artifact_io._write_arrays(path, {"probability_shift": np.array([.1])}, {})
    with pytest.raises(ValueError, match="rerun compare"):
        artifact_io.load_shift(path)
