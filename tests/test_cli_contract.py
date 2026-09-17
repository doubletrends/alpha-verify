from __future__ import annotations

from contextlib import redirect_stderr
from io import StringIO
import unittest

from alphaverify.cli import COMMANDS, build_parser
from alphaverify.infrastructure.workspace import STAGES


class CliContractTests(unittest.TestCase):
    def test_pipeline_cli_exposes_stage_subcommands(self) -> None:
        parser = build_parser()
        help_text = parser.format_help()

        for command in (
            "measure",
            "compare",
            "validate",
            "select",
            "forecast",
        ):
            self.assertIn(command, help_text)
        for removed in ("summarize", "status", "NODE"):
            self.assertNotIn(removed, help_text)
        for removed in ("--gate", "--family", "--rerun", "--read", "--fdr"):
            self.assertNotIn(removed, help_text)
        self.assertNotIn("--verbose", help_text)
        args = parser.parse_args(["measure", "--workspace", "btc_daily"])
        self.assertEqual(args.command, "measure")
        self.assertEqual(args.workspace, "btc_daily")
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["compare", "--verbose"])

    def test_commands_preserve_pipeline_order(self) -> None:
        self.assertEqual(
            [command.name for command in COMMANDS],
            ["measure", "compare", "validate", "select", "forecast"],
        )

    def test_stage_commands_mirror_the_stage_order(self) -> None:
        self.assertEqual(
            tuple(command.stage for command in COMMANDS), STAGES
        )

    def test_validation_has_no_stage_specific_options(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["validate", "--workspace", "btc_daily"])

        self.assertEqual(args.command, "validate")
        self.assertEqual(args.workspace, "btc_daily")
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["validate", "--fdr", "0.1"])


if __name__ == "__main__":
    unittest.main()
