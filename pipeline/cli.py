"""Command-line interface for the barrier-touch pipeline."""

from __future__ import annotations

import argparse

from pipeline.composition_command import cmd_bayes
from pipeline.inspect_commands import cmd_read, cmd_status
from pipeline.report_command import cmd_report
from pipeline.stat_commands import cmd_gate, cmd_validation
from pipeline.surface_commands import cmd_summary, cmd_surface
from workspace import Workspace


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Barrier-touch pipeline: P(price reaches theta within h | condition), "
            "measured on a full grid and judged on a coarse one."
        )
    )
    parser.add_argument("--workspace", metavar="NAME", default="nasdaq_daily")
    parser.add_argument("--family", metavar="NAME", help="Restrict to one family")
    parser.add_argument("--rerun", action="store_true", help="Rebuild artifacts that already exist")
    parser.add_argument(
        "--fdr",
        type=float,
        default=0.05,
        metavar="Q",
        help="Benjamini-Hochberg false-discovery rate for --gate (default 0.05)",
    )

    g = parser.add_mutually_exclusive_group()
    g.add_argument("--surface", action="store_true", help="1. write 01_surface_array and 01_surface_xlsx")
    g.add_argument(
        "--summary",
        action="store_true",
        help="2. write 02_summary_array and 02_summary_xlsx; runs the economic filter",
    )
    g.add_argument(
        "--validation",
        action="store_true",
        help="3. write 03_validation_array and 03_validation_xlsx",
    )
    g.add_argument(
        "--gate",
        action="store_true",
        help="4. correct across the sweep (BH) and intersect with the economic filter",
    )
    g.add_argument(
        "--bayes",
        action="store_true",
        help="5. compose conditions out of sample, 05_bayes.npz + 05_bayes.json",
    )
    g.add_argument("--report", action="store_true", help="6. render figures into workspace result/")
    g.add_argument("--status", action="store_true", help="Inventory by family")
    g.add_argument("--read", metavar="ID", help="Print a node surface summary")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    ws = Workspace(args.workspace)

    if args.surface:
        cmd_surface(ws, args.family, args.rerun)
    elif args.summary:
        cmd_summary(ws, args.family)
    elif args.validation:
        cmd_validation(ws, args.family, args.rerun)
    elif args.gate:
        cmd_gate(ws, args.fdr)
    elif args.bayes:
        cmd_bayes(ws)
    elif args.report:
        cmd_report(ws)
    elif args.status:
        cmd_status(ws)
    elif args.read:
        cmd_read(ws, args.read)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
