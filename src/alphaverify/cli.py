"""Command-line interface for the barrier-touch pipeline."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from alphaverify.domain import tensor_runtime
from alphaverify.infrastructure.scaffold import available_templates, copy_workspace
from alphaverify.infrastructure.workspace import (
    STAGE_DIRECTORIES,
    WORKSPACES_DIRNAME,
    Workspace,
    workspaces_root,
)
from alphaverify.pipeline.reporting import RULE, ansi_styles, color_enabled
from alphaverify.pipeline.step_01_surface import cmd_surface
from alphaverify.pipeline.step_02_shift import cmd_shift
from alphaverify.pipeline.step_03_validation import cmd_validation
from alphaverify.pipeline.step_04_selection import cmd_selection
from alphaverify.pipeline.step_05_forecast import cmd_forecast


@dataclass(frozen=True)
class Command:
    name: str
    summary: str
    handler: Callable[[Workspace], None]
    stage: str


COMMANDS = (
    Command(
        "measure",
        "Measure raw conditional probabilities",
        cmd_surface,
        stage="surface",
    ),
    Command(
        "compare",
        "Compare raw probabilities to baseline",
        cmd_shift,
        stage="shift",
    ),
    Command(
        "validate",
        "Validate all eligible bins vs null",
        cmd_validation,
        stage="validation",
    ),
    Command(
        "select",
        "Select cleared bins and heatmaps",
        cmd_selection,
        stage="selection",
    ),
    Command(
        "forecast",
        "Cleared nodes and their forecast",
        cmd_forecast,
        stage="forecast",
    ),
)
COMMAND_BY_NAME = {command.name: command for command in COMMANDS}

# Setup, not a stage: it writes the workspace that the stage commands then read.
INIT_COMMAND = "init"


def overview(color: bool) -> str:
    """Root help: setup, then the pipeline commands in order with their artifact directories."""
    accent, bold, dim, reset = ansi_styles(color)

    def heading(text: str) -> list[str]:
        return [f"  {accent}{bold}{text}{reset}", ""]

    def row(name: str, summary: str, target: str) -> str:
        return (
            f"    {bold}{name}{reset}{' ' * (12 - len(name))}"
            f"{summary:<38}→ {dim}{target}{reset}"
        )

    lines = [
        f"{accent}{RULE}{reset}",
        "AlphaVerify",
        "Conditional barrier-touch probability pipeline",
        f"{accent}{RULE}{reset}",
        "",
        *heading("Setup"),
        row(INIT_COMMAND, "Copy a shipped workspace here", f"{WORKSPACES_DIRNAME}/NAME/"),
        "",
        *heading("Pipeline"),
    ]
    for command in COMMANDS:
        lines.append(row(command.name, command.summary, f"{STAGE_DIRECTORIES[command.stage]}/"))
    lines += [
        "",
        *heading("Options"),
        "    --workspace NAME        Select a workspace",
        "    --workspaces-dir DIR    Workspace root (or $ALPHAVERIFY_WORKSPACES)",
        "    --cuda                  Use CUDA numerical kernels",
        "    -h, --help              Show this help",
    ]
    return "\n".join(lines) + "\n"


class RootParser(argparse.ArgumentParser):
    """Root help is the command overview; subcommands keep argparse option help."""

    def format_help(self) -> str:
        return overview(color_enabled())


def _add_workspace_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--workspace", metavar="NAME", default="nasdaq_daily")
    parser.add_argument(
        "--workspaces-dir",
        metavar="DIR",
        default=None,
        help=(
            "directory holding workspaces "
            f"(default: $ALPHAVERIFY_WORKSPACES, else ./{WORKSPACES_DIRNAME})"
        ),
    )


def _add_run_options(parser: argparse.ArgumentParser) -> None:
    _add_workspace_options(parser)
    parser.add_argument(
        "--cuda",
        action="store_true",
        help="run numerical barrier kernels on CUDA (requires an available CUDA PyTorch device)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = RootParser(prog="alphaverify")
    commands = parser.add_subparsers(
        dest="command", metavar="COMMAND", parser_class=argparse.ArgumentParser
    )

    _add_workspace_options(
        commands.add_parser(
            INIT_COMMAND,
            help=f"copy a shipped workspace ({', '.join(available_templates())}) into place",
        )
    )
    for command in COMMANDS:
        _add_run_options(commands.add_parser(command.name))
    return parser


def cmd_init(name: str, workspaces_dir: str | None) -> None:
    """Write one shipped workspace declaration where the stage commands will look for it."""
    root = workspaces_root(workspaces_dir)
    written = copy_workspace(name, root)
    accent, bold, dim, reset = ansi_styles(color_enabled())
    print(f"\n{accent}{RULE}{reset}")
    print(f"{accent}{bold}Workspace {name}{reset} → {root / name}")
    print(f"{accent}{RULE}{reset}\n")
    for path in written:
        print(f"  {dim}{path.relative_to(root)}{reset}")
    option = "" if root == Path.cwd() / WORKSPACES_DIRNAME else f" --workspaces-dir {root}"
    print(f"\n  next: alphaverify measure --workspace {name}{option}\n")


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return
    if args.command == INIT_COMMAND:
        cmd_init(args.workspace, args.workspaces_dir)
        return
    ws = Workspace(args.workspace, args.workspaces_dir)
    tensor_runtime.configure(args.cuda)

    COMMAND_BY_NAME[args.command].handler(ws)


if __name__ == "__main__":
    main()
