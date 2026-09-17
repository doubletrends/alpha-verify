"""Shared fixtures for downstream pipeline stages."""

import numpy as np
import pytest

from alphaverify.infrastructure import artifact_io
from alphaverify.infrastructure.workspace import STAGE_DIRECTORIES, Workspace
from alphaverify.pipeline import step_03_validation as validation_stage


@pytest.fixture
def selected_workspace(tmp_path):
    nodes = [
        {"id": node, "family": "test", "feature": "day_of_week", "params": {}}
        for node in ("a", "b")
    ]
    declaration = tmp_path / "workspaces" / "example" / "universe.json"
    artifact_io.write_json(declaration, {
        "meta": {"asset": {"ticker": "TEST", "interval": "1d"}, "start_date": "2024-01-01",
                 "min_obs": 100, "n_bins": 2, "barriers": {"min": -.20, "max": .20, "step": .01},
                 "horizons": {"min": 1, "max": 30}, "evaluate": {"min_dev": .10, "min_bin_n": 50, "min_run": 2}},
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
        surface = {
            "conditional_probability": probability,
            "bin_hit_counts": np.full_like(probability, 40, dtype=int),
            "bin_observation_counts": np.full((2, 2), 80),
            "eligible_observation_count": np.array([160, 158]),
            "barriers": deltas, "horizons": horizons, "bin_edges": np.array([.5]),
            "bin_assignments": np.array([0, 1], dtype=np.uint8),
            "index": np.array(["2024-01-01", "2024-01-02"]),
            "open": np.array([100., 101.]), "high": np.array([101., 102.]),
            "low": np.array([99., 100.]), "close": np.array([100., 101.]),
            "volume": np.ones(2), "feature_values": np.array([0., 1.]),
        }
        meta = {"node": node["id"], "bin_labels": ["x < 0.5", "0.5 < x"]}
        artifact_io.save_surface(surface, ws.cube_path(node["id"]), meta)
        artifact_io.save_shift(
            probability - baseline[:, None, :] + offset / 100, ws.shift_cube_path(node["id"]), meta,
        )
    artifact_io.save_surface({
        **surface, "conditional_probability": baseline[:, None, :],
        "bin_hit_counts": surface["bin_hit_counts"][:, :1],
        "bin_observation_counts": surface["bin_observation_counts"][:1],
        "bin_edges": np.array([]), "bin_assignments": np.zeros(2, dtype=np.uint8),
    }, ws.baseline_cube, {"node": "baseline", "bin_labels": ["all"]})
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
        "workspace": "example", "artifact": STAGE_DIRECTORIES["validation"], "complete": True,
        "method": validation_stage.METHOD,
        "input_fingerprint": validation_stage.input_fingerprint(ws),
        "missing_nodes": [], "summary": {"nodes": 2, "tested": 2, "skipped": 0, "cleared": 1},
        "tests": rows, "skipped_bins": [],
        "cleared": [{key: value for key, value in rows[0].items() if key != "null_scores"}],
    }
    artifact_io.write_json(ws.validation_summary_path, summary)
    return ws

