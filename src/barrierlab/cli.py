"""Command-line interface for the barrier-touch pipeline."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass

from barrierlab.infrastructure.artifacts import STAGE_DIRECTORIES
from barrierlab.infrastructure.workspace import Workspace
from barrierlab.pipeline.step_01_surface import cmd_surface
from barrierlab.pipeline.step_02_shift import cmd_shift
from barrierlab.pipeline.step_03_selection import cmd_selection
from barrierlab.pipeline.step_04_validation import cmd_validation
from barrierlab.pipeline.step_05_redundancy import cmd_redundancy
from barrierlab.pipeline.step_06_composition import cmd_composition
from barrierlab.pipeline.step_07_report import cmd_report
from barrierlab.pipeline.status import cmd_status


@dataclass(frozen=True)
class Command:
    name: str
    help: str
    handler: Callable[..., None]
    stage_directory: str | None = None
    accepts_node: bool = False


COMMANDS = (
    Command(
        "surface",
        "1. write surface arrays and workbooks",
        cmd_surface,
        STAGE_DIRECTORIES["surface"],
    ),
    Command(
        "shift",
        "2. write baseline-subtracted arrays and workbooks",
        cmd_shift,
        STAGE_DIRECTORIES["shift"],
    ),
    Command(
        "selection",
        "3. rank and retain the strongest nodes",
        cmd_selection,
        STAGE_DIRECTORIES["selection"],
    ),
    Command(
        "validation",
        "4. validate selected nodes and write final verdicts",
        cmd_validation,
        STAGE_DIRECTORIES["validation"],
    ),
    Command(
        "redundancy",
        "5. map dependence among cleared nodes",
        cmd_redundancy,
        STAGE_DIRECTORIES["redundancy"],
    ),
    Command(
        "composition",
        "6. evaluate out-of-sample composition and current forecasts",
        cmd_composition,
        STAGE_DIRECTORIES["composition"],
    ),
    Command(
        "report",
        "7. render audience-facing figures",
        cmd_report,
        STAGE_DIRECTORIES["report"],
    ),
    Command(
        "status",
        "show workspace or node artifact and validation status",
        cmd_status,
        accepts_node=True,
    ),
)
COMMAND_BY_NAME = {command.name: command for command in COMMANDS}


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

    for command in COMMANDS:
        subparser = commands.add_parser(command.name, help=command.help)
        if command.accepts_node:
            subparser.add_argument(
                "node",
                nargs="?",
                metavar="NODE",
                help="show detailed status for this node ID",
            )
        _add_workspace(subparser)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return
    ws = Workspace(args.workspace)

    command = COMMAND_BY_NAME[args.command]
    if command.accepts_node:
        command.handler(ws, args.node)
    else:
        command.handler(ws)


if __name__ == "__main__":
    main()
