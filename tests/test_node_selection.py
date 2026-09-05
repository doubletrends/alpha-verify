from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from domain.selection import (
    load_selected_node,
    rank_nodes,
    save_selected_node,
    score_node_information,
    selected_node_from_shift_cube,
)


def _cube(rates: list[float]) -> dict:
    """A one-target, ten-bin cube with 100 observations per bin."""
    n_bins = len(rates)
    hits = np.rint(np.asarray(rates) * 100).astype(int)
    return {
        "Δs": np.array([-0.10]),
        "horizons": np.array([14]),
        "bin_n": np.full((n_bins, 1), 100),
        "hits": hits[None, :, None],
        "prob": np.asarray(rates)[None, :, None],
        "base": np.array([[0.50]]),
        "shift": np.zeros((1, n_bins, 1)),
        "n_obs": np.array([n_bins * 100]),
        "edges": np.arange(1, n_bins, dtype=float),
        "meta": {"bin_labels": [f"bin {i + 1}" for i in range(n_bins)]},
    }


class NodeInformationSelectionTests(unittest.TestCase):
    def test_broad_gradient_can_outrank_an_extreme_only_effect(self) -> None:
        # Both tables average to the same 50% prior.  The isolated regime is much
        # stronger locally, but the small ordered differences across all ten bins carry
        # more expected information for a model that sees every bin.
        extreme = _cube([0.70] + [0.4777777778] * 9)
        broad = _cube([0.35, 0.38, 0.42, 0.46, 0.49, 0.51, 0.54, 0.58, 0.62, 0.65])

        extreme_score = score_node_information(extreme, -0.10, 14, shrink_k=0.0)
        broad_score = score_node_information(broad, -0.10, 14, shrink_k=0.0)

        self.assertIsNotNone(extreme_score)
        self.assertIsNotNone(broad_score)
        self.assertGreater(broad_score["score"], extreme_score["score"])
        self.assertGreater(broad_score["effective_bins"], 5.5)
        self.assertLess(extreme_score["effective_bins"], 3.0)

    def test_representative_bin_is_the_largest_information_contributor(self) -> None:
        result = score_node_information(
            _cube([0.35, 0.38, 0.42, 0.46, 0.49, 0.51, 0.54, 0.58, 0.62, 0.65]),
            -0.10,
            14,
            shrink_k=0.0,
        )

        self.assertEqual(result["bin"], int(np.argmax(result["bin_information"])))
        self.assertAlmostEqual(result["score"], sum(result["bin_information"]))
        self.assertEqual(result["best_cell"]["horizon"], 14)

    def test_stage_three_has_no_economic_verdict(self) -> None:
        cube = _cube([0.35, 0.38, 0.42, 0.46, 0.49, 0.51, 0.54, 0.58, 0.62, 0.65])
        result = rank_nodes(
            [{"id": "example", "family": "test", "feature": "close", "params": {}}],
            lambda node: cube,
            top_k=1,
            delta=-0.10,
            horizon=14,
            shrink_k=0.0,
        )

        self.assertNotIn("economic", result["selected"][0])
        self.assertNotIn("economic", result["method"])

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
            path = Path(directory) / "selected.npz"
            save_selected_node(artifact, path, artifact["meta"])
            loaded = load_selected_node(path)

        np.testing.assert_array_equal(loaded["shift"], cube["shift"])
        self.assertEqual(loaded["shift"].shape[1], 4)


if __name__ == "__main__":
    unittest.main()
