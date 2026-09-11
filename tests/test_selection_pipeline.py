"""Stage 4 selects exactly validation-cleared bins and renders their heatmaps."""

from unittest.mock import Mock

import numpy as np
import pytest

from alphaverify.infrastructure import artifact_io
from alphaverify.infrastructure.workspace import Workspace
from alphaverify.pipeline import step_03_validation as validation_stage
from alphaverify.pipeline import step_04_selection as selection_stage
from alphaverify.presentation import selection_plots


@pytest.fixture
def selected_workspace(tmp_path):
    nodes = [
        {"id": node, "family": "test", "feature": "day_of_week", "params": {}}
        for node in ("a", "b")
    ]
    declaration = tmp_path / "workspaces" / "example" / "universe.json"
    artifact_io.write_json(declaration, {
        "meta": {"asset": {"ticker": "TEST"}, "start_date": "2024-01-01", "n_bins": 2},
        "families": {"test": nodes},
    })
    ws = Workspace("example", tmp_path / "workspaces")
    deltas = np.array([-.02, 0., .02])
    horizons = np.array([1, 3])
    for offset, node in enumerate(nodes):
        probability = np.array([
            [[.20, .25], [.30, .35]],
            [[1.00, 1.00], [1.00, 1.00]],
            [[.45, .50], [.55, .60]],
        ])
        baseline = np.array([[.25, .30], [1., 1.], [.50, .55]])
        cube = {
            "prob": probability,
            "base": baseline,
            "probability_shift": probability - baseline[:, None, :] + offset / 100,
            "hits": np.full_like(probability, 40, dtype=int),
            "bin_n": np.full((2, 2), 80), "n_obs": np.array([160, 158]),
            "\u0394s": deltas, "horizons": horizons, "edges": np.array([.5]),
            "index": np.array(["2024-01-01", "2024-01-02"]),
            "high": np.array([101., 102.]), "low": np.array([99., 100.]),
            "close": np.array([100., 101.]), "feature_values": np.array([0., 1.]),
        }
        artifact_io.save_shift(cube, ws.shift_cube_path(node["id"]), {
            "node": node["id"], "bin_labels": ["x < 0.5", "0.5 < x"],
        })
    rows = [
        {"node": "a", "family": "test", "feature": "day_of_week", "params": {},
         "bin": 0, "bin_number": 1, "bin_label": "x < 0.5", "bin_score": .12,
         "monte_carlo_p_value": .01, "cleared": True, "null_p95": .10,
         "n_null_replicates": 1000, "n_supported_null": 1000},
        {"node": "b", "family": "test", "feature": "day_of_week", "params": {},
         "bin": 1, "bin_number": 2, "bin_label": "0.5 < x", "bin_score": .08,
         "monte_carlo_p_value": .20, "cleared": False, "null_p95": .10,
         "n_null_replicates": 1000, "n_supported_null": 1000},
    ]
    summary = {
        "workspace": "example", "artifact": "03_validation", "complete": True,
        "method": validation_stage._method(),
        "input_fingerprint": validation_stage.input_fingerprint(ws),
        "missing_nodes": [], "summary": {"nodes": 2, "tested": 2, "skipped": 0, "cleared": 1},
        "tests": rows, "skipped_bins": [], "cleared": [rows[0]],
    }
    ws.write_json(ws.validation_summary_path, summary)
    return ws


def test_selection_contains_every_and_only_cleared_bin(selected_workspace, monkeypatch):
    render = Mock(return_value=[])
    monkeypatch.setattr(selection_plots, "write_selected_shift_heatmaps", render)
    selection_stage.cmd_selection(selected_workspace)
    result = selected_workspace.read_json(selected_workspace.selection_summary_path)
    assert result["complete"]
    assert [(row["node"], row["bin"]) for row in result["selected"]] == [("a", 0)]
    assert result["summary"] == {"bins": 1, "nodes": 1}
    assert selection_stage.selection_summary_is_current(selected_workspace, result)


def test_selection_renders_full_shift_heatmap(selected_workspace):
    validation_plot = selected_workspace.validation_summary_path.parent / "plot"
    validation_plot.mkdir(parents=True)
    source_histogram = validation_plot / "null_histogram__a__bin_01.png"
    source_histogram.write_bytes(b"stage-three-distribution")
    selection_stage.cmd_selection(selected_workspace)
    plots = list((selected_workspace.selection_summary_path.parent / "plot").glob("*.png"))
    assert {path.name for path in plots} == {
        "selected_shift_heatmap__a__bin_01.png",
        "null_histogram__a__bin_01.png",
    }
    heatmap = next(path for path in plots if path.name.startswith("selected_shift_heatmap"))
    copied = next(path for path in plots if path.name.startswith("null_histogram"))
    assert heatmap.stat().st_size > 1000
    assert copied.read_bytes() == source_histogram.read_bytes()


def test_selection_becomes_stale_when_validation_changes(selected_workspace, monkeypatch):
    monkeypatch.setattr(selection_plots, "write_selected_shift_heatmaps", Mock(return_value=[]))
    selection_stage.cmd_selection(selected_workspace)
    result = selected_workspace.read_json(selected_workspace.selection_summary_path)
    validation = selected_workspace.read_json(selected_workspace.validation_summary_path)
    validation["generated"] = "changed"
    selected_workspace.write_json(selected_workspace.validation_summary_path, validation)
    assert not selection_stage.selection_summary_is_current(selected_workspace, result)
