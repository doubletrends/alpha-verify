"""Probability-difference units across computation, persistence, and display."""

import numpy as np
import pytest
from openpyxl import load_workbook

from alphaverify.domain import scoring, shift
from alphaverify.infrastructure.workspace import WorkspaceConfig
from alphaverify.presentation import workbooks

META = {
    "asset": {"ticker": "TEST", "interval": "1d"}, "start_date": "2024-01-01",
    "min_obs": 100, "n_bins": 10, "barriers": {"min": -.20, "max": .20, "step": .01},
    "horizons": {"min": 1, "max": 30}, "evaluate": {"min_dev": .10, "min_bin_n": 50, "min_run": 2},
}


def test_threshold_is_a_probability_difference():
    assert WorkspaceConfig.from_meta(META).min_dev == .10
    with pytest.raises(ValueError):
        WorkspaceConfig.from_meta({**META, "evaluate": {**META["evaluate"], "min_dev": 10}})


@pytest.mark.parametrize("field", sorted(META))
def test_every_meta_setting_must_be_declared(field):
    with pytest.raises(KeyError):
        WorkspaceConfig.from_meta({key: value for key, value in META.items() if key != field})


def test_shift_storage_threshold_and_workbook_share_probability_units(tmp_path):
    cube = {
        "conditional_probability": np.array([[[.25]], [[.75]]]),
        "bin_observation_counts": np.array([[100]]),
        "bin_hit_counts": np.array([[[25]], [[75]]]),
        "eligible_observation_count": 100,
        "barriers": np.array([.01, .02]), "horizons": np.array([1]),
        "bin_edges": np.array([]), "meta": {"bin_labels": ["all"]},
    }
    baseline = np.array([[.125], [.5]])
    result = {**cube, "probability_shift": shift.from_cube(cube, baseline),
              "baseline_probability": baseline}
    np.testing.assert_array_equal(result["probability_shift"], [[[.125]], [[.25]]])
    assert shift.strongest_cell(result, 0, min_dev=.125, min_bin_n=50, min_run=2) is not None
    assert shift.strongest_cell(result, 0, min_dev=.126, min_bin_n=50, min_run=2) is None
    path = tmp_path / "shift.xlsx"
    workbooks.write_shift_xlsx(result, path, "example", "d")
    book = load_workbook(path)
    sheet = book.active
    assert sheet.cell(5, 2).value == .25
    assert sheet.cell(6, 2).value == .125
    assert sheet.cell(5, 2).number_format == '+0.0%;-0.0%;0.0%'
    book.close()


def test_score_rescaling_preserves_comparisons_including_ties(monkeypatch):
    # Binary-exact probabilities include a tied null and a supported zero score.
    probability = np.array([
        [[[.25], [.5]], [[.75], [.5]]],
        [[[.25], [.5]], [[.75], [.5]]],
        [[[.375], [.5]], [[.625], [.5]]],
        [[[.125], [.5]], [[.875], [.5]]],
    ])
    baseline = np.full((4, 2, 1), .5)
    counts = np.full((4, 2, 1), 100)
    new = scoring.bin_scores(probability, baseline, counts, [-.1, .1])
    original = scoring.baseline_shifts
    monkeypatch.setattr(scoring, "baseline_shifts", lambda *args: 100 * original(*args))
    old = scoring.bin_scores(probability, baseline, counts, [-.1, .1])
    np.testing.assert_allclose(new.bin_score, old.bin_score / 100)
    np.testing.assert_array_equal(new.score_supported, old.score_supported)
    np.testing.assert_array_equal(new.bin_score[1:] >= new.bin_score[0],
                                  old.bin_score[1:] >= old.bin_score[0])


def test_strongest_cell_reports_the_requested_bin():
    # Bin 0 has the larger run; bin 1 must report its own best cell.
    shifts = np.array([[[.30], [.12]], [[.30], [.12]], [[0.], [0.]]])
    cube = {
        "probability_shift": shifts, "conditional_probability": shifts + .5,
        "baseline_probability": np.full((3, 1), .5), "bin_hit_counts": np.full((3, 2, 1), 40),
        "bin_observation_counts": np.full((2, 1), 100),
        "barriers": np.array([-.02, -.01, .01]), "horizons": np.array([5]),
    }
    assert shift.strongest_cell(cube, 0, .10, 50, 2)["dev"] == .30
    best = shift.strongest_cell(cube, 1, .10, 50, 2)
    assert (best["bin"], best["dev"]) == (1, .12)


def test_barrier_labels_are_exact_for_the_grid_step():
    from alphaverify.infrastructure.workspace import WorkspaceConfig
    from alphaverify.presentation.display import barrier_number_format, format_barrier

    def grid(step, bound):
        return WorkspaceConfig.from_meta(
            {**META, "barriers": {"min": -bound, "max": bound, "step": step}}
        ).barriers

    whole, quarter = grid(.01, .2), grid(.0025, .03)
    assert [format_barrier(value, whole) for value in (-.2, 0., .07)] == ["-20%", "+0%", "+7%"]
    assert barrier_number_format(whole) == "+0%;-0%;0%"
    labels = [format_barrier(value, quarter) for value in quarter]
    assert labels[11:14] == ["-0.25%", "+0.00%", "+0.25%"] and len(set(labels)) == len(quarter)
    assert barrier_number_format(quarter) == "+0.00%;-0.00%;0%"
    assert format_barrier(.002, grid(.002, .03)) == "+0.2%"
