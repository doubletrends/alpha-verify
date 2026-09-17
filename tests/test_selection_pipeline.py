"""Stage 4 selects exactly validation-cleared bins and renders their figures."""

from unittest.mock import Mock

import numpy as np
import pytest

from alphaverify.infrastructure import artifact_io
from alphaverify.infrastructure.workspace import Workspace
from alphaverify.pipeline import step_03_validation as validation_stage
from alphaverify.pipeline import step_04_selection as selection_stage
from alphaverify.presentation import bin_figures


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
         "n_null_replicates": 1000, "n_supported_null": 1000,
         "null_scores": np.linspace(0, .11, 1000).tolist()},
        {"node": "b", "family": "test", "feature": "day_of_week", "params": {},
         "bin": 1, "bin_number": 2, "bin_label": "0.5 < x", "bin_score": .08,
         "monte_carlo_p_value": .20, "cleared": False, "null_p95": .10,
         "n_null_replicates": 1000, "n_supported_null": 1000,
         "null_scores": np.linspace(0, .11, 1000).tolist()},
    ]
    summary = {
        "workspace": "example", "artifact": "03_validation", "complete": True,
        "method": validation_stage._method(),
        "input_fingerprint": validation_stage.input_fingerprint(ws),
        "missing_nodes": [], "summary": {"nodes": 2, "tested": 2, "skipped": 0, "cleared": 1},
        "tests": rows, "skipped_bins": [],
        "cleared": [{key: value for key, value in rows[0].items() if key != "null_scores"}],
    }
    artifact_io.write_json(ws.validation_summary_path, summary)
    return ws


def test_selection_contains_every_and_only_cleared_bin(selected_workspace, monkeypatch):
    render = Mock(return_value=[])
    monkeypatch.setattr(bin_figures, "write_bin_figures", render)
    selection_stage.cmd_selection(selected_workspace)
    result = artifact_io.read_json(selected_workspace.selection_summary_path)
    assert result["complete"]
    assert [(row["node"], row["bin"]) for row in result["selected"]] == [("a", 0)]
    assert result["summary"] == {"bins": 1, "nodes": 1}
    assert "null_scores" not in result["selected"][0]
    rendered_rows = render.call_args.args[1]
    assert len(rendered_rows[0]["null_scores"]) == 1000
    assert selection_stage.selection_summary_is_current(selected_workspace, result)


def test_selection_renders_heatmap_beside_null_distribution(selected_workspace, monkeypatch):
    plot_dir = selected_workspace.selection_summary_path.parent / "plot"
    plot_dir.mkdir(parents=True)
    # Views from former layouts are replaced, not left beside the standard figure.
    for stale in ("selected_shift_heatmap__a__bin_01.png", "null_histogram__a__bin_01.png",
                  "selected_bin__a__bin_01.png"):
        (plot_dir / stale).write_bytes(b"old")
    panels = []
    original_subplots = bin_figures.plt.subplots

    def capture(*args, **kwargs):
        fig, axes = original_subplots(*args, **kwargs)
        panels.append(axes)
        return fig, axes

    monkeypatch.setattr(bin_figures.plt, "subplots", capture)
    selection_stage.cmd_selection(selected_workspace)
    plots = list(plot_dir.glob("*.png"))
    assert [path.name for path in plots] == ["bin_figure__a__bin_01.png"]
    assert plots[0].stat().st_size > 1000
    heatmap_ax, null_ax = panels[0]
    figure = heatmap_ax.figure
    # Each panel owns half of the grid; the colorbar is carved out of the heatmap's half.
    heatmap_half = heatmap_ax.get_subplotspec().get_topmost_subplotspec().get_position(figure)
    null_half = null_ax.get_subplotspec().get_topmost_subplotspec().get_position(figure)
    assert heatmap_half.width == pytest.approx(null_half.width)
    colorbar_ax = next(ax for ax in figure.axes if ax not in (heatmap_ax, null_ax))
    assert colorbar_ax.get_position().x1 <= heatmap_half.x1 + 1e-9
    assert len(null_ax.patches) > 0 and len(heatmap_ax.collections) == 1


def test_selection_becomes_stale_when_validation_changes(selected_workspace, monkeypatch):
    monkeypatch.setattr(bin_figures, "write_bin_figures", Mock(return_value=[]))
    selection_stage.cmd_selection(selected_workspace)
    result = artifact_io.read_json(selected_workspace.selection_summary_path)
    validation = artifact_io.read_json(selected_workspace.validation_summary_path)
    validation["generated"] = "changed"
    artifact_io.write_json(selected_workspace.validation_summary_path, validation)
    assert not selection_stage.selection_summary_is_current(selected_workspace, result)
