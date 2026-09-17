"""Stage 5 lists cleared nodes, combines active cleared bins one per family, and checks them against history."""

from collections import Counter

import numpy as np
import pandas as pd
import pytest
import torch

from alphaverify.domain import barrier, combination
from alphaverify.infrastructure import artifact_io
from alphaverify.infrastructure.artifact_history import feature_bins_from_artifact
from alphaverify.infrastructure.workspace import Workspace
from alphaverify.pipeline import step_02_shift
from alphaverify.pipeline import step_05_forecast as forecast_stage

BARRIERS, HORIZONS = np.array([-.02, -.01, .01, .02]), np.array([1, 5])
FEATURES = {  # node: (family, feature values over time)
    "a": ("osc", lambda t: np.sin(t / 7)),
    "b": ("osc", lambda t: np.sin(t / 7 + .2)),
    "c": ("mom", lambda t: np.cos(t / 23)),
}


@pytest.fixture
def forecast_workspace(tmp_path, monkeypatch):
    nodes = [{"id": "baseline", "family": "_base", "feature": "constant", "params": {}}] + [
        {"id": node, "family": family, "feature": node, "params": {}}
        for node, (family, _) in FEATURES.items()
    ]
    artifact_io.write_json(tmp_path / "workspaces" / "example" / "universe.json", {
        "meta": {"asset": {"ticker": "TEST", "interval": "1d"}, "start_date": "2024-01-01",
                 "min_obs": 100, "n_bins": 4, "barriers": {"min": -.02, "max": .02, "step": .01},
                 "horizons": {"min": 1, "max": 5}, "evaluate": {"min_dev": .10, "min_bin_n": 50, "min_run": 2}},
        "families": {"all": nodes},
    })
    ws = Workspace("example", tmp_path / "workspaces")
    rng = np.random.default_rng(9)
    n = 1200
    close = 100 * np.exp(np.cumsum(rng.normal(0, .015, n)))
    data = pd.DataFrame({"open": close, "high": close * 1.008, "low": close * .992,
                         "close": close, "volume": np.ones(n)},
                        index=pd.date_range("2020-01-01", periods=n))
    t = np.arange(n)
    for node in nodes:
        values = np.zeros(n) if node["id"] == "baseline" else FEATURES[node["id"]][1](t)
        feature = pd.Series(values, index=data.index)
        edges = np.array([]) if node["id"] == "baseline" else barrier.bin_edges(feature, 4)
        cube = barrier.touch_tensor(data, feature, HORIZONS, BARRIERS, edges)
        cube.update({key: data[key].to_numpy() for key in data},
                    index=data.index.astype(str).to_numpy(), feature_values=values)
        artifact_io.save_surface(cube, ws.cube_path(node["id"]), {
            "node": node["id"], "bin_labels": barrier.bin_labels(edges),
        })
    step_02_shift.cmd_shift(ws)

    last_bins = {node: int(feature_bins_from_artifact(
        artifact_io.load_surface(ws.cube_path(node)))[-1]) for node in FEATURES}
    inactive_c = (last_bins["c"] + 1) % 4

    def selected(node, b, rank, p, q):
        return {"node": node, "family": FEATURES[node][0], "feature": node, "params": {},
                "bin": b, "bin_number": b + 1, "bin_label": f"bin {b + 1}",
                "rank": rank, "monte_carlo_p_value": p, "q_value": q}

    artifact_io.write_json(ws.validation_summary_path, {
        "tests": [{"node": node, "bin": b} for node in FEATURES for b in range(4)],
    })
    artifact_io.write_json(ws.selection_summary_path, {
        "summary": {"expected_by_chance": .6},
        "selected": [
            selected("a", last_bins["a"], 1, .001, .01),
            selected("b", last_bins["b"], 2, .002, .01),
            selected("c", last_bins["c"], 3, .003, .01),
            selected("c", inactive_c, 4, .004, .01),
        ],
    })
    monkeypatch.setattr(forecast_stage, "selection_summary_is_current", lambda ws, selection: True)
    ws.last_bins = last_bins
    return ws


def test_cleared_nodes_follow_their_strongest_bin_and_group_bins_in_order():
    def row(node, b, rank):
        return {"node": node, "family": "test", "feature": "day_of_week", "params": {},
                "bin": b, "bin_number": b + 1, "bin_label": f"bin {b + 1}",
                "rank": rank, "monte_carlo_p_value": rank / 100, "q_value": rank / 50}

    # Node a's strongest bin is its second one.
    selected = [row("a", 1, 1), row("a", 0, 2), row("b", 0, 2)]
    nodes = forecast_stage.cleared_nodes(selected, Counter(a=2, b=2, c=2))
    assert [(node["node"], node["bins_cleared"], node["bins_tested"], node["best_rank"],
             [item["bin"] for item in node["bins"]]) for node in nodes] == [
        ("a", 2, 2, 1, [0, 1]), ("b", 1, 2, 2, [0]),
    ]


def test_forecast_uses_one_active_bin_per_family(forecast_workspace):
    ws = forecast_workspace
    forecast_stage.cmd_forecast(ws)
    result = artifact_io.read_json(ws.forecast_path)

    # a and b share a family; c's inactive second bin is not a condition.
    assert {row["node"]: row["used"] for row in result["conditions"]} == {"a": True, "b": False, "c": True}


def test_forecast_surfaces_match_stage1_counts_and_an_independent_joint_measurement(forecast_workspace):
    ws = forecast_workspace
    forecast_stage.cmd_forecast(ws)
    result = artifact_io.read_json(ws.forecast_path)
    surface = {node: artifact_io.load_surface(ws.cube_path(node)) for node in ("baseline", "a", "c")}
    used = [("a", ws.last_bins["a"]), ("c", ws.last_bins["c"])]

    expected, expected_counts = combination.naive_bayes_probability(
        surface["baseline"]["bin_hit_counts"][:, 0, :], surface["baseline"]["bin_observation_counts"][0],
        [surface[node]["bin_hit_counts"][:, b, :] for node, b in used],
        [surface[node]["bin_observation_counts"][b] for node, b in used],
    )
    combined = np.array(result["combined_probability"], dtype=float)
    np.testing.assert_allclose(combined, expected, rtol=0, atol=1e-12)
    assert result["combined_observation_counts"] == expected_counts.astype(int).tolist()

    # Measure "every used condition held" as its own Stage 1 feature; bin 1 is the joint rate.
    both = np.all([feature_bins_from_artifact(surface[node]) == b for node, b in used], axis=0)
    data = pd.DataFrame({key: surface["baseline"][key] for key in ("open", "high", "low", "close", "volume")})
    joint_cube = barrier.touch_tensor(data, pd.Series(both.astype(float)), HORIZONS, BARRIERS, np.array([.5]))
    joint = np.array(result["joint_probability"], dtype=float)
    np.testing.assert_allclose(joint, joint_cube["conditional_probability"][:, 1, :], rtol=0, atol=1e-12)
    assert result["joint_observation_counts"] == joint_cube["bin_observation_counts"][1].tolist()
    assert np.isfinite(joint).any()


def test_no_cleared_bins_forecasts_the_baseline(forecast_workspace):
    ws = forecast_workspace
    selection = artifact_io.read_json(ws.selection_summary_path)
    selection["selected"] = []
    artifact_io.write_json(ws.selection_summary_path, selection)
    forecast_stage.cmd_forecast(ws)
    result = artifact_io.read_json(ws.forecast_path)

    np.testing.assert_allclose(
        np.array(result["combined_probability"], dtype=float),
        np.array(result["baseline_probability"], dtype=float), rtol=0, atol=1e-12,
    )


def test_forecast_requires_a_current_selection(selected_workspace):
    forecast_stage.cmd_forecast(selected_workspace)
    assert not selected_workspace.forecast_path.exists()


def logit(p):
    return np.log(p / (1 - p))


def test_two_conditions_add_their_log_odds_ratios():
    hits, counts = np.array([[20.]]), np.array([98.])
    first, second = (np.array([[40.]]), np.array([58.])), (np.array([[9.]]), np.array([48.]))
    base, p1, p2 = 21 / 100, 41 / 60, 10 / 50
    expected = 1 / (1 + np.exp(-(logit(base) + (logit(p1) - logit(base)) + (logit(p2) - logit(base)))))
    actual, support = combination.naive_bayes_probability(
        hits, counts, [first[0], second[0]], [first[1], second[1]]
    )
    np.testing.assert_allclose(actual, [[expected]])
    np.testing.assert_array_equal(support, [48.])


def test_thin_baseline_or_condition_cells_are_unsupported():
    hits, counts = np.zeros((1, 2)), np.array([100., 29.])
    result, support = combination.naive_bayes_probability(
        hits, counts, [np.zeros((1, 2))], [np.array([29., 100.])]
    )
    assert np.isnan(result).all()
    np.testing.assert_array_equal(support, [29., 29.])


def test_feature_bins_use_stored_edges_the_stage1_tie_convention_and_missing_values():
    edges = np.array([.5, 1.5])
    values = np.array([.2, .5, 1.0, 1.5, 9.0, np.nan])
    bins = feature_bins_from_artifact({"feature_values": values, "bin_edges": edges})
    assert bins.tolist() == [0, 0, 1, 1, 2, -1]
    expected = barrier.bin_indices(torch.from_numpy(values[None, :5]), torch.from_numpy(edges[None]))
    assert bins[:5].tolist() == expected[0].tolist()
