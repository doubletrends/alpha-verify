"""Read-only status reporting for pipeline artifacts."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from barrierlab.domain import shift
from barrierlab.infrastructure import artifact_io
from barrierlab.infrastructure.workspace import Workspace
from barrierlab.pipeline.step_03_validation import validation_summary_is_current
from barrierlab.pipeline.step_04_selection import selection_summary_is_current
from barrierlab.pipeline.context import materialized_shift


def _print_node_status(ws: Workspace, node_id: str) -> None:
    """Print one node's strongest shift and its available null result."""
    node = ws.catalog.find(node_id)
    path = ws.shift_cube_path(node_id)
    if not path.exists():
        print(f"No shift array for {node_id} - run the compare command first.")
        return

    cube = materialized_shift(ws, node_id)
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


def cmd_status(ws: Workspace, node_id: str | None = None) -> None:
    if node_id is not None:
        _print_node_status(ws, node_id)
        return

    validation = ws.read_json(ws.validation_summary_path)
    validation_current = validation_summary_is_current(ws, validation)
    cleared_rows = validation.get("cleared", []) if validation_current else []
    tests = validation.get("tests", []) if validation_current else []
    selection = ws.read_json(ws.selection_summary_path)
    selection_current = selection_summary_is_current(ws, selection)
    selected_rows = selection.get("selected", []) if selection_current else []

    columns = [
        "nodes",
        "cube",
        "sheet",
        "shift",
        "sheet",
        "tested",
        "cleared",
        "selected",
    ]
    by_family = defaultdict(lambda: [0] * len(columns))
    for node in ws.catalog.all_nodes():
        counts = by_family[node["family"]]
        counts[0] += 1
        counts[1] += bool(ws.has_cube(node["id"]))
        counts[2] += bool(ws.has_surface(node["id"]))
        counts[3] += bool(ws.has_shift_cube(node["id"]))
        counts[4] += bool(ws.has_shift_surface(node["id"]))
        counts[5] += sum(1 for row in tests if row.get("node") == node["id"])
        counts[6] += sum(1 for row in cleared_rows if row.get("node") == node["id"])
        counts[7] += sum(1 for row in selected_rows if row.get("node") == node["id"])

    print(f"\n=== Status [{ws.dir.name}] ===")
    print(
        f"  Δ {ws.deltas[0]:+.0%}..{ws.deltas[-1]:+.0%}   "
        f"horizons +{ws.horizons[0]}{ws.horizon_unit}.."
        f"+{ws.horizons[-1]}{ws.horizon_unit}   {ws.n_bins} bins"
    )
    if validation and validation_current:
        method = validation.get("method", {})
        print(
            f"  validation: {len(tests)} condition bins vs {method.get('null')} | "
            f"raw p < {method['threshold']['raw_p']} cleared {len(cleared_rows)}"
        )
    elif validation:
        missing = validation.get("missing_nodes", [])
        detail = f" ({len(missing)} missing nodes)" if missing else ""
        print(f"  validation: incomplete or stale{detail} - run validate")
    if selection and selection_current:
        print(f"  selection: {len(selected_rows)} cleared bins rendered in Stage 4")
    elif selection:
        print("  selection: stale - run select")

    print()
    header = f"  {'family':<15}" + "".join(
        f"{name:>9}" for name in columns
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for family in sorted(by_family):
        print(
            f"  {family:<15}"
            + "".join(f"{value:>9}" for value in by_family[family])
        )
    totals = [
        sum(by_family[family][index] for family in by_family)
        for index in range(len(columns))
    ]
    print("  " + "-" * (len(header) - 2))
    print(f"  {'TOTAL':<15}" + "".join(f"{value:>9}" for value in totals))
    print(
        f"\n  full grid {len(ws.deltas)}Δ x {len(ws.horizons)}t   |   "
        "03_validation tests all eligible condition bins across available nodes"
    )
