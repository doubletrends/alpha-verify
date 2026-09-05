from __future__ import annotations

import unittest

from cli import build_parser
from scripts.read_node import build_parser as build_read_node_parser


class CliContractTests(unittest.TestCase):
    def test_pipeline_cli_exposes_only_numbered_stage_actions(self) -> None:
        help_text = build_parser().format_help()

        for action in (
            "--surface",
            "--shift",
            "--selection",
            "--validation",
            "--redundancy",
            "--bayes",
            "--report",
        ):
            self.assertIn(action, help_text)
        for removed in ("--gate", "--status", "--read"):
            self.assertNotIn(removed, help_text)

    def test_node_reader_has_a_separate_cli(self) -> None:
        args = build_read_node_parser().parse_args(
            ["vix_level", "--workspace", "nasdaq_daily"]
        )

        self.assertEqual(args.node, "vix_level")
        self.assertEqual(args.workspace, "nasdaq_daily")


if __name__ == "__main__":
    unittest.main()
