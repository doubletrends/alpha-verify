"""Read-only status reporting for pipeline artifacts."""

from __future__ import annotations

from collections import defaultdict

from infrastructure.workspaces.workspace import Workspace
from pipeline.step_04_validation import (
    validation_artifact_is_current,
    validation_summary_is_current,
)
from pipeline.step_05_redundancy import redundancy_artifacts_are_current


def cmd_status(ws: Workspace) -> None:
    selection = ws.read_json(ws.selection_path)
    selected = [
        row for row in selection.get("selected", []) if ws.has_selection_array(row)
    ]
    validation = ws.read_json(ws.validation_summary_path)
    redundancy = ws.read_json(ws.redundancy_path)
    validation_current = validation_summary_is_current(ws, validation)
    cleared_rows = validation.get("cleared", []) if validation_current else []
    economic_nodes = (
        {
            row["node"]
            for row in validation.get("economics", [])
            if row.get("passed")
        }
        if validation_current
        else set()
    )
    tests = validation.get("tests", []) if validation_current else []
    null_nodes = {row["node"] for row in tests if row.get("null_pass")}
    fdr_nodes = {row["node"] for row in tests if row.get("fdr_pass")}

    columns = [
        "nodes",
        "cube",
        "sheet",
        "shift",
        "sheet",
        "select",
        "s-sheet",
        "valid",
        "v-sheet",
        "econ",
        "cleared",
    ]
    by_family = defaultdict(lambda: [0] * len(columns))
    for node in ws.catalog.all_nodes():
        counts = by_family[node["family"]]
        counts[0] += 1
        counts[1] += bool(ws.has_cube(node["family"], node["id"]))
        counts[2] += bool(ws.has_surface(node["family"], node["id"]))
        counts[3] += bool(ws.has_shift_cube(node["family"], node["id"]))
        counts[4] += bool(ws.has_shift_surface(node["family"], node["id"]))
        selected_rows = [row for row in selected if row["node"] == node["id"]]
        counts[5] += sum(1 for row in selected_rows if ws.has_selection_array(row))
        counts[6] += sum(1 for row in selected_rows if ws.has_selection_surface(row))
        valid_rows = [
            row for row in selected_rows if validation_artifact_is_current(ws, row)
        ]
        counts[7] += len(valid_rows)
        counts[8] += sum(1 for row in valid_rows if ws.has_validation_surface(row))
        counts[9] += sum(
            1 for row in selected_rows if row["node"] in economic_nodes
        )
        counts[10] += sum(
            1 for row in cleared_rows if row.get("node") == node["id"]
        )

    print(f"\n=== Status [{ws.dir.name}] ===")
    print(
        f"  Δ {ws.deltas[0]:+.0%}..{ws.deltas[-1]:+.0%}   "
        f"horizons +{ws.horizons[0]}{ws.horizon_unit}.."
        f"+{ws.horizons[-1]}{ws.horizon_unit}   {ws.n_bins} bins"
    )
    if validation and validation_current:
        method = validation.get("method", {})
        print(
            f"  validation: raw p<={method.get('null_alpha')} {len(null_nodes)} nodes | "
            f"BH q={method.get('q')} {len(fdr_nodes)} | economic {len(economic_nodes)} | "
            f"all three {len(cleared_rows)}"
        )
    elif validation:
        missing = validation.get("missing_nodes", [])
        detail = f" ({len(missing)} missing nodes)" if missing else ""
        print(f"  validation: incomplete or stale{detail} - run validation")

    if redundancy_artifacts_are_current(ws, validation, redundancy):
        print(
            f"  redundancy: {len(redundancy.get('nodes', []))} cleared nodes in "
            f"{len(redundancy.get('clusters', []))} conditional-dependence clusters"
        )
    elif redundancy:
        print("  redundancy: stale for current validation - run redundancy")

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
        f"03_selection keeps top {len(selected)} nodes and one representative bin each"
    )
