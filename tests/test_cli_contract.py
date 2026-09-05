from __future__ import annotations

from contextlib import redirect_stderr
from io import StringIO
import unittest

from cli import build_parser
from scripts.read_node import build_parser as build_read_node_parser


class CliContractTests(unittest.TestCase):
    def test_pipeline_cli_exposes_stage_and_status_subcommands(self) -> None:
        parser = build_parser()
        help_text = parser.format_help()

        for command in (
            "surface",
            "shift",
            "selection",
            "validation",
            "redundancy",
            "bayes",
            "report",
            "status",
        ):
            self.assertIn(command, help_text)
        for removed in ("--gate", "--family", "--rerun", "--read", "--fdr"):
            self.assertNotIn(removed, help_text)

        args = parser.parse_args(["surface", "--workspace", "btc_daily"])
        self.assertEqual(args.command, "surface")
        self.assertEqual(args.workspace, "btc_daily")

    def test_validation_has_no_stage_specific_options(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["validation", "--workspace", "btc_daily"])

        self.assertEqual(args.command, "validation")
        self.assertEqual(args.workspace, "btc_daily")
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["validation", "--fdr", "0.1"])

    def test_node_reader_has_a_separate_cli(self) -> None:
        args = build_read_node_parser().parse_args(
            ["vix_level", "--workspace", "nasdaq_daily"]
        )

        self.assertEqual(args.node, "vix_level")
        self.assertEqual(args.workspace, "nasdaq_daily")


if __name__ == "__main__":
    unittest.main()
