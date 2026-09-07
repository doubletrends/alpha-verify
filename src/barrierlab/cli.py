"""Command-line interface for the barrier-touch pipeline."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass

from barrierlab.domain import barrier
from barrierlab.infrastructure.workspace import Workspace
from barrierlab.pipeline.step_01_surface import cmd_surface
from barrierlab.pipeline.step_02_shift import cmd_shift
from barrierlab.pipeline.step_03_selection import cmd_selection
from barrierlab.pipeline.step_04_validation import cmd_validation
from barrierlab.pipeline.status import cmd_status


@dataclass(frozen=True)
class Command:
    name: str
    help: str
    handler: Callable[..., None]
    accepts_node: bool = False


COMMANDS = (
    Command(
        "measure",
        "1. calculate conditional probability surfaces",
        cmd_surface,
    ),
    Command(
        "compare",
        "2. calculate baseline-relative probability shifts",
        cmd_shift,
    ),
    Command(
        "select",
        "3. rank and retain the strongest condition effects",
        cmd_selection,
    ),
    Command(
        "validate",
        "4. test selected effects and write final verdicts",
        cmd_validation,
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
    parser.add_argument(
        "--cuda",
        action="store_true",
        help="run numerical barrier kernels on CUDA (requires an available CUDA PyTorch device)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="show per-node progress and diagnostic details",
    )


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
    barrier.configure_cuda(args.cuda)

    command = COMMAND_BY_NAME[args.command]
    if command.accepts_node:
        command.handler(ws, args.node)
    else:
        command.handler(ws, verbose=args.verbose)


if __name__ == "__main__":
    main()
