"""Command-line interface for the barrier-touch pipeline."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
import sys
from textwrap import dedent

from barrierlab.domain import barrier
from barrierlab.infrastructure.workspace import Workspace
from barrierlab.pipeline.step_01_surface import cmd_surface
from barrierlab.pipeline.step_02_shift import cmd_shift
from barrierlab.pipeline.step_03_validation import cmd_validation
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
        "1. measure raw conditional probabilities",
        cmd_surface,
    ),
    Command(
        "compare",
        "2. compare raw probabilities to the baseline",
        cmd_shift,
    ),
    Command(
        "validate",
        "3. validate all eligible condition bins against the null",
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

OVERVIEW = dedent("""\
    ============================================================
    BarrierLab
    Conditional barrier-touch probability pipeline
    ============================================================

      Pipeline

        measure     Measure raw conditional probabilities → 01_surface/
        compare     Compare raw probabilities to baseline → 02_shift/
        validate    Validate all eligible bins vs null     → 03_validation/

      Inspect

        status [NODE]    Show workspace or node results

      Options

        --workspace NAME    Select a workspace
        --cuda              Use CUDA numerical kernels
        -h, --help          Show this help
    """)


class RootParser(argparse.ArgumentParser):
    """Keep the root command overview distinct from subcommand option help."""

    def format_help(self) -> str:
        if not sys.stdout.isatty():
            return OVERVIEW

        cyan, bold, dim, reset = "\033[36m", "\033[1m", "\033[2m", "\033[0m"
        help_text = OVERVIEW.replace("=" * 60, f"{cyan}{'=' * 60}{reset}")
        for heading in ("Pipeline", "Inspect", "Options"):
            help_text = help_text.replace(f"  {heading}", f"  {cyan}{bold}{heading}{reset}")
        for command, artifact in (
            ("measure", "01_surface/"),
            ("compare", "02_shift/"),
            ("validate", "03_validation/"),
        ):
            help_text = help_text.replace(
                f"    {command}", f"    {bold}{command}{reset}"
            ).replace(f"→ {artifact}", f"→ {dim}{artifact}{reset}")
        help_text = help_text.replace("    status", f"    {bold}status{reset}")
        return help_text


def _add_workspace(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--workspace", metavar="NAME", default="nasdaq_daily")
    parser.add_argument(
        "--cuda",
        action="store_true",
        help="run numerical barrier kernels on CUDA (requires an available CUDA PyTorch device)",
    )
def build_parser() -> argparse.ArgumentParser:
    parser = RootParser(
        description=(
            "Barrier-touch pipeline: P(price reaches Δ within t | condition), "
            "measured on a full grid and judged after subtracting the baseline."
        )
    )
    commands = parser.add_subparsers(
        dest="command", metavar="COMMAND", parser_class=argparse.ArgumentParser
    )

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
        command.handler(ws)


if __name__ == "__main__":
    main()
