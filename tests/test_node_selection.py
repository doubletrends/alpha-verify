from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from barrierlab.domain.selection import (
    rank_nodes,
    bin_information,
    selected_node_from_shift_cube,
)
from barrierlab.infrastructure import artifact_io


def _cube(
    positive_rates: list[float],
    negative_rates: list[float] | None = None,
) -> dict:
    """A two-sided, one-horizon cube with 100 observations per bin."""
    if negative_rates is None:
        negative_rates = [0.50] * len(positive_rates)
    n_bins = len(positive_rates)
    assert len(negative_rates) == n_bins
    hits = np.rint(np.asarray([negative_rates, positive_rates]) * 100).astype(int)
    prob = np.asarray([negative_rates, positive_rates])[:, :, None]
    base = np.array([[0.50], [0.50]])
    return {
        "Δs": np.array([-0.10, 0.10]),
        "horizons": np.array([14]),
        "bin_n": np.full((n_bins, 1), 100),
        "hits": hits[:, :, None],
        "prob": prob,
        "base": base,
        "shift": (prob - base[:, None, :]) * 100.0,
        "n_obs": np.array([n_bins * 100]),
        "edges": np.arange(1, n_bins, dtype=float),
        "meta": {"bin_labels": [f"bin {i + 1}" for i in range(n_bins)]},
    }


class NodeStrongestBinSelectionTests(unittest.TestCase):
    def test_strongest_bin_score_is_used_for_its_selection_row(self) -> None:
        # Each bin is independently ranked by its positive-versus-negative shift skew.
        extreme = _cube([0.70] + [0.4777777778] * 9)
        broad = _cube([0.35, 0.38, 0.42, 0.46, 0.49, 0.51, 0.54, 0.58, 0.62, 0.65])

        extreme_score = bin_information(extreme, -0.10, 14)
        broad_score = bin_information(broad, -0.10, 14)

        self.assertIsNotNone(extreme_score)
        self.assertIsNotNone(broad_score)
        self.assertGreater(extreme_score["score"], broad_score["score"])
        self.assertAlmostEqual(extreme_score["score_pp"], 20.0)

    def test_representative_bin_is_the_largest_positive_negative_skew(self) -> None:
        result = bin_information(
            _cube(
                [0.60, 0.55, 0.50],
                [0.45, 0.50, 0.30],
            ),
            -0.10,
            14,
        )

        self.assertEqual(result["bin"], int(np.argmax(result["bin_score"])))
        self.assertAlmostEqual(result["score"], max(result["bin_score"]))
        self.assertAlmostEqual(result["best_cell"]["positive_shift"], 0.0)
        self.assertAlmostEqual(result["best_cell"]["negative_shift"], -20.0)
        self.assertEqual(result["best_cell"]["horizon"], 14)

    def test_stage_three_has_no_economic_verdict(self) -> None:
        cube = _cube([0.35, 0.38, 0.42, 0.46, 0.49, 0.51, 0.54, 0.58, 0.62, 0.65])
        result = rank_nodes(
            [{"id": "example", "family": "test", "feature": "close", "params": {}}],
            lambda node: cube,
            top_k=1,
            delta=-0.10,
            horizon=14,
        )

        self.assertNotIn("economic", result["selected"][0])
        self.assertNotIn("economic", result["method"])
        self.assertNotIn("generated", result)

    def test_selected_node_artifact_preserves_every_bin(self) -> None:
        cube = _cube([0.35, 0.45, 0.55, 0.65])
        row = {"bin": 2, "rank": 1, "score": 0.02, "bin_label": "bin 3"}

        artifact = selected_node_from_shift_cube(cube, row)

        np.testing.assert_array_equal(artifact["shift"], cube["shift"])
        np.testing.assert_array_equal(artifact["prob"], cube["prob"])
        np.testing.assert_array_equal(artifact["bin_n"], cube["bin_n"])
        self.assertEqual(int(artifact["source_bin"]), 2)
        self.assertEqual(artifact["meta"]["bin_labels"], cube["meta"]["bin_labels"])

        with TemporaryDirectory() as directory:
            path = Path(directory) / "selected.safetensors"
            artifact_io.save_selected_node(artifact, path, artifact["meta"])
            loaded = artifact_io.load_selected_node(path)

        np.testing.assert_allclose(loaded["shift"], cube["shift"])
        self.assertEqual(loaded["shift"].shape[1], 4)


if __name__ == "__main__":
    unittest.main()
