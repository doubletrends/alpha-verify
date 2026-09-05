"""Stage 4: selected-sheet validation nulls."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from domain import shift, validation as val
from infrastructure.workspaces.workspace import Workspace
from pipeline.context import artifact_node_feature
from presentation import workbooks


def _validation_matches_selection(ws: Workspace, row: dict) -> bool:
    if not ws.has_validation_array(row):
        return False
    try:
        grid = shift.load(ws.shift_cube_path(row["family"], row["node"]))
        existing = val.load(ws.validation_array_path(row))
    except Exception:
        return False
    return (
        np.array_equal(existing["Δs"], grid["Δs"])
        and np.array_equal(existing["horizons"], grid["horizons"])
        and existing["cell_real"].shape == (grid["shift"].shape[0], 1, grid["shift"].shape[2])
        and int(np.asarray(existing.get("source_bin", -1))) == int(row["bin"])
        and "sheet_peak_p" in existing
    )


def _selected_rows(ws: Workspace, family: str | None = None) -> list[dict]:
    selection = ws.read_json(ws.selection_path)
    rows = selection.get("selected", [])
    if family:
        rows = [row for row in rows if row.get("family") == family]
    return rows


def cmd_validation(ws: Workspace, family: str | None = None, rerun: bool = False) -> None:
    """Run null tests for selected bin sheets and render their workbooks."""
    rows = _selected_rows(ws, family)
    if not rows:
        print("No selected sheets - run --selection first.")
        return
    rows = [row for row in rows if ws.has_selection_array(row)]
    if not rows:
        print("No 03_selection_array artifacts - run --selection first.")
        return
    nodes = {node["id"]: node for node in ws.catalog.all_nodes()}
    rows = [
        row for row in rows
        if row["node"] in nodes and ws.has_shift_cube(row["family"], row["node"])
    ]
    if not rerun:
        rows = [
            row for row in rows
            if not _validation_matches_selection(ws, row)
            or not ws.has_validation_surface(row)
        ]
    if not rows:
        print("Nothing to validate or render (use --rerun to redo selected sheets).")
        return

    deltas, horizons = ws.shift_deltas, ws.shift_horizons
    print(f"\n=== 4. 04_validation_array [{ws.dir.name}] - {len(rows)} selected sheets ===")
    print(
        f"  selected sheets are tested as {len(deltas)} delta x {len(horizons)} horizon surfaces; "
        "validation artifacts are selected-bin scoped"
    )
    print("  sheet p-values use the peak over that bin's 2D surface, not the whole node cube")
    print(f"  guard {val.EDGE_GUARD} bars either side, so the p-value floor is 1/(usable+1)")
    print("  nulls the strongest touch-probability deviation per horizon\n")

    array_count = 0
    workbook_count = 0
    node_cache: dict[str, dict] = {}
    result_cache: dict[str, dict] = {}
    for row in rows:
        node = nodes[row["node"]]
        try:
            if node["id"] not in node_cache:
                node_cache[node["id"]] = shift.load(
                    ws.shift_cube_path(node["family"], node["id"])
                )
            cube = node_cache[node["id"]]
            if node["id"] not in result_cache:
                data, feature = artifact_node_feature(cube, ws.min_obs)
                result_cache[node["id"]] = val.validate_node(
                    data, feature, cube["horizons"], cube["Δs"], cube["edges"]
                )
            result = val.sheet_from_node_result(result_cache[node["id"]], row)
            val.save(result, ws.validation_array_path(row), {
                "node": node["id"],
                "family": node["family"],
                "feature": node["feature"],
                "params": node["params"],
                "workspace": ws.dir.name,
                "bin_labels": result["meta"]["bin_labels"],
                "source_bin": int(row["bin"]),
                "source_bin_number": int(row["bin_number"]),
                "selection_rank": int(row["rank"]),
                "selection_score": float(row["score"]),
                "generated": datetime.now(timezone.utc).isoformat(),
            })
            array_count += 1
            workbooks.write_validation_xlsx(
                result, ws.validation_surface_path(row), node["id"], node["feature"],
                node["params"], ws.horizon_unit,
            )
            workbook_count += 1
        except Exception as error:
            print(f"  {node['id']:<26} [skip] {error}")
            continue

        source_bin = int(row["bin"])
        sheet_p = result["sheet_peak_p"]
        finite = np.isfinite(sheet_p)
        if finite.any():
            _, horizon_index = np.unravel_index(
                int(np.nanargmin(np.where(finite, sheet_p, np.nan))), sheet_p.shape
            )
            floor = 1.0 / (1.0 + result["n_shifts"][horizon_index])
            at_floor = result["sheet_peak_p"][0, horizon_index] <= floor + 1e-12
            tag = "  [at the floor]" if at_floor else ""
            print(
                f"  {node['id']:<26} bin {source_bin + 1:>2} "
                f"best p={result['sheet_peak_p'][0, horizon_index]:.5f} "
                f"at +{result['horizons'][horizon_index]}{ws.horizon_unit}  peak "
                f"{result['sheet_peak_real'][0, horizon_index]:5.1f} vs p95 "
                f"{result['sheet_peak_p95'][0, horizon_index]:5.1f}{tag}"
            )
        else:
            print(f"  {node['id']:<26} insufficient data in selected bins")

    relative_dir = ws.dir.relative_to(ws.root_dir)
    print(f"\n  wrote {array_count} .npz artifacts under {relative_dir}/04_validation_array/")
    print(f"  wrote {workbook_count} workbooks under {relative_dir}/04_validation_xlsx/")
    print("  verdicts are assigned by --gate across the selected sheet/horizon tests")
