from __future__ import annotations

from contextlib import redirect_stderr
from io import StringIO
import unittest
from unittest.mock import patch

from cli import COMMANDS, build_parser
from pipeline.status import cmd_status


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
            "composition",
            "report",
            "status",
        ):
            self.assertIn(command, help_text)
        for removed in ("--gate", "--family", "--rerun", "--read", "--fdr"):
            self.assertNotIn(removed, help_text)
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["bayes"])

        args = parser.parse_args(["surface", "--workspace", "btc_daily"])
        self.assertEqual(args.command, "surface")
        self.assertEqual(args.workspace, "btc_daily")

    def test_numbered_commands_match_their_stage_directories(self) -> None:
        numbered = [command for command in COMMANDS if command.stage_directory]

        self.assertEqual(len(numbered), 7)
        for number, command in enumerate(numbered, start=1):
            prefix, name = command.stage_directory.split("_", 1)
            self.assertEqual(int(prefix), number)
            self.assertEqual(name, command.name)

    def test_validation_has_no_stage_specific_options(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["validation", "--workspace", "btc_daily"])

        self.assertEqual(args.command, "validation")
        self.assertEqual(args.workspace, "btc_daily")
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["validation", "--fdr", "0.1"])

    def test_status_accepts_an_optional_node(self) -> None:
        parser = build_parser()
        workspace_args = parser.parse_args(["status", "--workspace", "btc_daily"])
        node_args = parser.parse_args(
            ["status", "vix_level", "--workspace", "nasdaq_daily"]
        )

        self.assertIsNone(workspace_args.node)
        self.assertEqual(workspace_args.workspace, "btc_daily")
        self.assertEqual(node_args.node, "vix_level")
        self.assertEqual(node_args.workspace, "nasdaq_daily")

    def test_status_routes_a_node_to_the_detailed_view(self) -> None:
        workspace = object()

        with patch("pipeline.status._print_node_status") as print_node_status:
            cmd_status(workspace, "vix_level")

        print_node_status.assert_called_once_with(workspace, "vix_level")


if __name__ == "__main__":
    unittest.main()
