"""Print one node's strongest baseline-relative shift surface.

This is a read-only research utility, separate from the numbered pipeline CLI.

Example:
    python scripts/read_node.py vix_level --workspace nasdaq_daily
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from domain import shift, validation as val  # noqa: E402
from infrastructure.workspace import Workspace  # noqa: E402
from pipeline.step_04_validation import validation_artifact_is_current  # noqa: E402


def read_node(ws: Workspace, node_id: str) -> None:
    """Print qualifying rows from one node's shift cube and its available null result."""
    node = ws.catalog.find(node_id)
    path = ws.shift_cube_path(node["family"], node_id)
    if not path.exists():
        print(f"No shift array for {node_id} - run the shift command first.")
        return

    cube = shift.load(path)
    dev = cube["shift"]
    deltas = cube["Δs"]
    horizons = cube["horizons"]
    labels = cube["meta"]["bin_labels"]

    print()
    print(f"{node_id}  [{node['family']}]  {node['feature']}  params={node['params']}")

    evaluation = shift.evaluate(cube, ws.min_dev, ws.min_bin_n, ws.min_run)
    best = evaluation["best"]
    if not best:
        print(f"  no cell reaches {ws.min_dev}pp across {ws.min_run} adjacent Δ rows")
        return

    bin_index = best["bin"]
    print(f"  strongest bin : {bin_index + 1} of {len(labels)}   ({labels[bin_index]})")
    print(
        f"  strongest cell: shift={best['dev']:+.1f}pp; P={best['prob']:.1%} "
        f"vs {best['base']:.1%} baseline "
        f"at Δ={best['Δ']:+.0%}, +{best['horizon']}{ws.horizon_unit} "
        f"(run={best['run']}, n={best['bin_n']})"
    )

    selection = ws.read_json(ws.selection_path)
    selected = [row for row in selection.get("selected", []) if ws.has_selection_array(row)]
    selected_row = next(
        (
            row for row in selected
            if row["node"] == node_id and int(row["bin"]) == int(bin_index)
        ),
        None,
    )
    if selected_row and validation_artifact_is_current(ws, selected_row):
        validation = val.load(ws.validation_array_path(selected_row))
        horizon_index = int(
            np.flatnonzero(validation["horizons"] == best["horizon"])[0]
        )
        floor = 1.0 / (1.0 + validation["n_shifts"][horizon_index])
        cell_p = validation["cell_p"][:, 0, horizon_index]
        peak_real = (
            validation["sheet_peak_real"][0, horizon_index]
            if "sheet_peak_real" in validation
            else validation["peak_real"][horizon_index]
        )
        peak_p95 = (
            validation["sheet_peak_p95"][0, horizon_index]
            if "sheet_peak_p95" in validation
            else validation["peak_p95"][horizon_index]
        )
        peak_p = (
            validation["sheet_peak_p"][0, horizon_index]
            if "sheet_peak_p" in validation
            else validation["peak_p"][horizon_index]
        )
        tag = " (at the floor)" if peak_p <= floor + 1e-12 else ""
        print(
            f"  null          : sheet peak {peak_real:.1f} vs p95 {peak_p95:.1f}, "
            f"p={peak_p:.5f}{tag}   floor {floor:.2e}"
        )
        print(
            f"                  {int(np.nansum(cell_p <= 0.05))} of "
            f"{int(np.isfinite(cell_p).sum())} cells in this bin are pointwise p<=0.05"
        )
    else:
        print("  null          : not validated yet")

    available = {int(value) for value in horizons}
    shown_horizons = [
        value for value in (1, 2, 3, 5, 7, 10, 14, 21, 30) if value in available
    ]
    columns = [int(np.flatnonzero(horizons == value)[0]) for value in shown_horizons]
    percent_decimals = 1 if ws.delta_step < 0.01 else 0
    delta_format = f"+.{percent_decimals}%"

    print()
    header = "".join(
        f"{'+' + str(value) + ws.horizon_unit:>8}" for value in shown_horizons
    )
    print(f"  {'Δ':>7}{header}")
    print("  " + "-" * (7 + 8 * len(shown_horizons)))
    shown = 0
    for index in sorted(range(len(deltas)), key=lambda item: -deltas[item]):
        row = dev[index, bin_index, columns]
        finite = row[np.isfinite(row)]
        if finite.size == 0 or np.max(np.abs(finite)) < ws.min_dev:
            continue
        cells = "".join(
            "       -" if not np.isfinite(value) else f"{value:>8.1f}"
            for value in row
        )
        print(f"  {format(deltas[index], delta_format):>7}{cells}")
        shown += 1
    if not shown:
        print(f"  no Δ row deviates by {ws.min_dev}pp at these horizons")
    print()
    print(
        "  values are percentage-point shifts from baseline; rows shown deviate "
        f"at least {ws.min_dev}pp"
    )
    print(
        f"  the workbook carries all {len(deltas)} Δ levels, {len(horizons)} horizons "
        f"and {len(labels)} bins"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Print one node's strongest baseline-relative shift surface."
    )
    parser.add_argument("node", metavar="ID")
    parser.add_argument("--workspace", metavar="NAME", default="nasdaq_daily")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    read_node(Workspace(args.workspace, ROOT / "workspaces"), args.node)


if __name__ == "__main__":
    main()
