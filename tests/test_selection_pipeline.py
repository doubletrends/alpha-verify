"""Stage 4 selects exactly validation-cleared bins, ranked by evidence, with FDR q-values."""

from unittest.mock import Mock

import numpy as np
import pytest

from alphaverify.domain.multiple_testing import benjamini_hochberg
from alphaverify.infrastructure import artifact_io
from alphaverify.pipeline import step_04_selection as selection_stage
from alphaverify.presentation import bin_figures


def test_ranking_uses_evidence_only_with_q_values_across_all_tests(selected_workspace, monkeypatch):
    monkeypatch.setattr(bin_figures, "write_bin_figures", Mock(return_value=[]))
    validation = artifact_io.read_json(selected_workspace.validation_summary_path)
    template = validation["tests"][0]
    # b/0 ties a/0 on p with a much larger score; a/1 has the strongest evidence and the smallest score.
    validation["tests"] += [
        {**template, "node": "b", "bin": 0, "bin_score": .90},
        {**template, "node": "a", "bin": 1, "bin_number": 2, "bin_score": .05,
         "monte_carlo_p_value": .004},
    ]
    validation["cleared"] = [
        {key: value for key, value in row.items() if key != "null_scores"}
        for row in validation["tests"] if row["cleared"]
    ]
    artifact_io.write_json(selected_workspace.validation_summary_path, validation)

    selection_stage.cmd_selection(selected_workspace)
    result = artifact_io.read_json(selected_workspace.selection_summary_path)
    ranked = [(row["node"], row["bin"], row["rank"]) for row in result["selected"]]
    assert ranked == [("a", 1, 1), ("a", 0, 2), ("b", 0, 2)]
    # Sorted p over all four tests: .004 .01 .01 .20 -> q = .04/3 for the three cleared bins.
    assert [row["q_value"] for row in result["selected"]] == pytest.approx([.04 / 3] * 3)
    assert result["summary"]["tested"] == 4
    assert result["summary"]["expected_by_chance"] == pytest.approx(.2)


def test_selection_becomes_stale_when_validation_changes(selected_workspace, monkeypatch):
    monkeypatch.setattr(bin_figures, "write_bin_figures", Mock(return_value=[]))
    selection_stage.cmd_selection(selected_workspace)
    result = artifact_io.read_json(selected_workspace.selection_summary_path)
    validation = artifact_io.read_json(selected_workspace.validation_summary_path)
    validation["generated"] = "changed"
    artifact_io.write_json(selected_workspace.validation_summary_path, validation)
    assert not selection_stage.selection_summary_is_current(selected_workspace, result)


def reference_q_values(p_values):
    """Step-up definition: q_i = min over p_(j) >= p_i of m * p_(j) / j, capped at 1."""
    ordered = sorted(p_values)
    m = len(ordered)
    return [
        min(1.0, min(m * ordered[j] / (j + 1) for j in range(m) if ordered[j] >= p))
        for p in p_values
    ]


def test_hand_worked_example_in_input_order():
    # Sorted p: .01 .02 .03 .20 -> m*p/j = .04 .04 .04 .20
    np.testing.assert_allclose(benjamini_hochberg([.20, .01, .03, .02]), [.20, .04, .04, .04])


def test_matches_step_up_definition_with_ties_and_the_monte_carlo_floor():
    rng = np.random.default_rng(5)
    p = np.round(rng.uniform(0, 1, 300) ** 3, 3)
    p[:12] = 1 / 1001
    rng.shuffle(p)
    np.testing.assert_allclose(benjamini_hochberg(p), reference_q_values(p.tolist()), rtol=0, atol=1e-12)


def test_no_tests_have_no_q_values():
    assert benjamini_hochberg([]).shape == (0,)
