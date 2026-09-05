"""Command-line interface for the barrier-touch pipeline."""

from __future__ import annotations

import argparse

from infrastructure.workspace import Workspace
from pipeline.step_01_surface import cmd_surface
from pipeline.step_02_shift import cmd_shift
from pipeline.step_03_selection import cmd_selection
from pipeline.step_04_validation import cmd_validation
from pipeline.step_05_redundancy import cmd_redundancy
from pipeline.step_06_composition import cmd_bayes
from pipeline.step_07_report import cmd_report
from pipeline.status import cmd_status


def _add_workspace(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--workspace", metavar="NAME", default="nasdaq_daily")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Barrier-touch pipeline: P(price reaches Δ within t | condition), "
            "measured on a full grid and judged after subtracting the baseline."
        )
    )
    commands = parser.add_subparsers(dest="command", metavar="COMMAND")

    surface = commands.add_parser("surface", help="1. write 01_surface artifacts")
    _add_workspace(surface)

    shift = commands.add_parser("shift", help="2. write 02_shift artifacts")
    _add_workspace(shift)

    selection = commands.add_parser("selection", help="3. write 03_selection artifacts")
    _add_workspace(selection)

    validation = commands.add_parser(
        "validation",
        help="4. validate selected nodes and write final verdicts",
    )
    _add_workspace(validation)

    redundancy = commands.add_parser(
        "redundancy",
        help="5. write redundancy NPZ, workbook, and manifest for cleared nodes",
    )
    _add_workspace(redundancy)

    bayes = commands.add_parser(
        "bayes",
        help="6. evaluate weighted Bayes and write current probability and shift workbooks",
    )
    _add_workspace(bayes)

    report = commands.add_parser("report", help="7. render figures into workspace result/")
    _add_workspace(report)

    status = commands.add_parser("status", help="show artifact and validation status")
    _add_workspace(status)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return
    ws = Workspace(args.workspace)

    if args.command == "surface":
        cmd_surface(ws)
    elif args.command == "shift":
        cmd_shift(ws)
    elif args.command == "selection":
        cmd_selection(ws)
    elif args.command == "validation":
        cmd_validation(ws)
    elif args.command == "redundancy":
        cmd_redundancy(ws)
    elif args.command == "bayes":
        cmd_bayes(ws)
    elif args.command == "report":
        cmd_report(ws)
    elif args.command == "status":
        cmd_status(ws)


if __name__ == "__main__":
    main()
