"""Command-line interface for the barrier-touch pipeline."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass

from alphaverify.domain import tensor_runtime
from alphaverify.infrastructure.workspace import STAGE_DIRECTORIES, Workspace
from alphaverify.pipeline.reporting import RULE, ansi_styles, color_enabled
from alphaverify.pipeline.step_01_surface import cmd_surface
from alphaverify.pipeline.step_02_shift import cmd_shift
from alphaverify.pipeline.step_03_validation import cmd_validation
from alphaverify.pipeline.step_04_selection import cmd_selection
from alphaverify.pipeline.step_05_summary import cmd_summary
from alphaverify.pipeline.step_06_forecast import cmd_forecast
from alphaverify.pipeline.status import cmd_status


@dataclass(frozen=True)
class Command:
    name: str
    help: str
    summary: str
    handler: Callable[..., None]
    stage: str | None = None
    accepts_node: bool = False


COMMANDS = (
    Command(
        "measure",
        "1. measure raw conditional probabilities",
        "Measure raw conditional probabilities",
        cmd_surface,
        stage="surface",
    ),
    Command(
        "compare",
        "2. compare raw probabilities to the baseline",
        "Compare raw probabilities to baseline",
        cmd_shift,
        stage="shift",
    ),
    Command(
        "validate",
        "3. validate all eligible condition bins against the null",
        "Validate all eligible bins vs null",
        cmd_validation,
        stage="validation",
    ),
    Command(
        "select",
        "4. retain validation-cleared bins and render heatmaps",
        "Select cleared bins and heatmaps",
        cmd_selection,
        stage="selection",
    ),
    Command(
        "summarize",
        "5. summarize which nodes and bins cleared",
        "Summarize cleared nodes and bins",
        cmd_summary,
        stage="summary",
    ),
    Command(
        "forecast",
        "6. combine cleared bins active on the last stored bar",
        "Forecast from active cleared bins",
        cmd_forecast,
        stage="forecast",
    ),
    Command(
        "status",
        "show workspace or node artifact and validation status",
        "Show workspace or node results",
        cmd_status,
        accepts_node=True,
    ),
)
COMMAND_BY_NAME = {command.name: command for command in COMMANDS}


def overview(color: bool) -> str:
    """Root help: the pipeline commands in order, each with its artifact directory."""
    accent, bold, dim, reset = ansi_styles(color)

    def heading(text: str) -> list[str]:
        return [f"  {accent}{bold}{text}{reset}", ""]

    lines = [
        f"{accent}{RULE}{reset}",
        "AlphaVerify",
        "Conditional barrier-touch probability pipeline",
        f"{accent}{RULE}{reset}",
        "",
        *heading("Pipeline"),
    ]
    for command in COMMANDS:
        if command.stage:
            lines.append(
                f"    {bold}{command.name}{reset}{' ' * (12 - len(command.name))}"
                f"{command.summary:<38}→ {dim}{STAGE_DIRECTORIES[command.stage]}/{reset}"
            )
    lines += ["", *heading("Inspect")]
    for command in COMMANDS:
        if not command.stage:
            usage = " [NODE]" if command.accepts_node else ""
            lines.append(
                f"    {bold}{command.name}{reset}{usage}"
                f"{' ' * (17 - len(command.name + usage))}{command.summary}"
            )
    lines += [
        "",
        *heading("Options"),
        "    --workspace NAME    Select a workspace",
        "    --cuda              Use CUDA numerical kernels",
        "    -h, --help          Show this help",
    ]
    return "\n".join(lines) + "\n"


class RootParser(argparse.ArgumentParser):
    """Keep the root command overview distinct from subcommand option help."""

    def format_help(self) -> str:
        return overview(color_enabled())


def _add_workspace(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--workspace", metavar="NAME", default="nasdaq_daily")
    parser.add_argument(
        "--cuda",
        action="store_true",
        help="run numerical barrier kernels on CUDA (requires an available CUDA PyTorch device)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = RootParser(
        prog="alphaverify",
        description=(
            "Barrier-touch pipeline: conditional probability by barrier, bin, and horizon, "
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
    tensor_runtime.configure(args.cuda)

    command = COMMAND_BY_NAME[args.command]
    if command.accepts_node:
        command.handler(ws, args.node)
    else:
        command.handler(ws)


if __name__ == "__main__":
    main()
