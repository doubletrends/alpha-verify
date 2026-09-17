"""Shared fixtures for downstream pipeline stages."""

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


@pytest.fixture
def summarized_workspace(selected_workspace, monkeypatch):
    """Nodes a and b clear bins; node c is tested but never clears."""
    monkeypatch.setattr(bin_figures, "write_bin_figures", Mock(return_value=[]))
    validation = artifact_io.read_json(selected_workspace.validation_summary_path)
    template = validation["tests"][0]
    validation["tests"] += [
        {**template, "node": "b", "bin": 0, "bin_label": "x < 0.5"},
        {**template, "node": "a", "bin": 1, "bin_number": 2, "bin_label": "0.5 < x",
         "monte_carlo_p_value": .004},
        {**template, "node": "c", "family": "other", "monte_carlo_p_value": .60, "cleared": False},
    ]
    validation["cleared"] = [
        {key: value for key, value in row.items() if key != "null_scores"}
        for row in validation["tests"] if row["cleared"]
    ]
    artifact_io.write_json(selected_workspace.validation_summary_path, validation)
    selection_stage.cmd_selection(selected_workspace)
    return selected_workspace
