from __future__ import annotations

import unittest

import numpy as np

from scripts.probe_expectancy import exhaust_surface, first_hit_probabilities


class ExpectancyProbeTests(unittest.TestCase):
    def test_competing_risk_masses_are_exhaustive(self) -> None:
        masses = first_hit_probabilities(
            np.array([0.20, 0.35, 0.50]),
            np.array([0.10, 0.20, 0.30]),
        )

        self.assertAlmostEqual(sum(masses), 1.0)
        self.assertTrue(all(0.0 <= value <= 1.0 for value in masses))

    def test_tie_policies_bound_the_expectancy(self) -> None:
        rows = exhaust_surface(
            np.array([
                [0.10, 0.20],
                [0.95, 0.99],
                [0.30, 0.50],
            ]),
            np.array([-0.10, 0.0, 0.10]),
            np.array([1, 2]),
            cost_bps=10,
        )

        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertLessEqual(
                row["expectancy_conservative"], row["expectancy_midpoint"]
            )
            self.assertLessEqual(
                row["expectancy_midpoint"], row["expectancy_optimistic"]
            )
            self.assertAlmostEqual(
                row["p_tp_first"]
                + row["p_sl_first"]
                + row["p_same_day"]
                + row["p_timeout"],
                1.0,
            )


if __name__ == "__main__":
    unittest.main()
