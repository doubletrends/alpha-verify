"""Stage 3 command: select nodes by their strongest-bin probability skew."""

from __future__ import annotations

from datetime import datetime, timezone
import shutil
from pathlib import Path

from openpyxl import load_workbook

from barrierlab.domain import selection
from barrierlab.infrastructure import artifact_io
from barrierlab.infrastructure.workspace import BASELINE_NODE, Workspace


def _copy_selected_workbook_sheet(source: Path, target: Path, bin_index: int) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    wb = load_workbook(target)
    keep = wb.sheetnames[int(bin_index)]
    for sheet_name in list(wb.sheetnames):
        if sheet_name != keep:
            del wb[sheet_name]
    wb.save(target)
    wb.close()


def _clear_selection_artifacts(ws: Workspace) -> None:
    """Discard generated Stage 3 views before replacing the selection manifest."""
    for file_type in ("array", "spreadsheet", "plot"):
        directory = ws.selection_path.parent / file_type
        if not directory.exists():
            continue
        for path in directory.iterdir():
            if path.is_file():
                path.unlink()


def cmd_selection(ws: Workspace) -> None:
    """Stage 3: retain nodes with the largest single-bin probability skew."""
    nodes = [
        n for n in ws.catalog.all_nodes()
        if n["id"] != BASELINE_NODE and ws.has_shift_cube(n["family"], n["id"])
    ]
    if not nodes:
        print("No shift arrays to select from - run shift first.")
        return

    def load_cube(node: dict) -> dict:
        return artifact_io.load_shift(ws.shift_cube_path(node["family"], node["id"]))

    baseline_path = ws.shift_cube_path("_base", BASELINE_NODE)
    if not baseline_path.exists():
        print("No baseline shift array - run shift first.")
        return
    delta, horizon = ws.composition_target(artifact_io.load_shift(baseline_path)["base"])
    result = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "artifact": "03_selection",
        "source": "02_shift",
        **selection.rank_nodes(
            nodes,
            load_cube,
            ws.selection_top_k,
            delta,
            horizon,
        ),
    }
    result["workspace"] = ws.dir.name
    result["families"] = sorted({n["family"] for n in nodes})

    selected = result["selected"]
    _clear_selection_artifacts(ws)
    copied_npz = 0
    copied_xlsx = 0
    by_id = {n["id"]: n for n in nodes}
    for row in selected:
        node = by_id[row["node"]]
        source = load_cube(node)
        selected_node = selection.selected_node_from_shift_cube(source, row)
        meta = {
            **selected_node["meta"],
            "artifact": "03_selection",
            "source": str(ws.shift_cube_path(row["family"], row["node"]).relative_to(ws.root_dir)),
            "node": row["node"],
            "family": row["family"],
            "feature": row["feature"],
            "params": row["params"],
            "selection": {
                "rank": row["rank"],
                "score": row["score"],
                "method": result["method"]["score"],
            },
        }
        artifact_io.save_selected_node(selected_node, ws.selection_array_path(row), meta)
        copied_npz += 1

        source_xlsx = ws.shift_surface_path(node["family"], node["id"])
        target_xlsx = ws.selection_surface_path(row)
        try:
            _copy_selected_workbook_sheet(source_xlsx, target_xlsx, int(row["bin"]))
            copied_xlsx += 1
        except PermissionError:
            print(f"  rank {row['rank']:>3} {row['node']:<26} [locked] close it in Excel and re-run")
        except FileNotFoundError:
            print(f"  rank {row['rank']:>3} {row['node']:<26} [no shift workbook] run shift to render it")

    ws.write_json(ws.selection_path, result)
    from barrierlab.presentation.report_nodes import write_selected_shift_graphs
    write_selected_shift_graphs(ws, selected)

    print(f"\n=== 3. Selection [{ws.dir.name}] ===")
    print(
        f"  scored {len(result['candidates'])} bins from 02_shift; "
        f"selected top {len(selected)} bins by absolute skew"
    )
    print(
        f"  target = P(touch ±{abs(delta):.0%} within {horizon}{ws.horizon_unit})  |  "
        "score = |02_shift(+Δ, bin) - 02_shift(-Δ, bin)|\n"
    )
    if selected:
        print(f"  {'rank':>4}  {'node':<26}{'family':<13}{'bin':>5}  {'skew':>7}  representative bin")
        print(f"  {'-'*4}  {'-'*26}{'-'*13}{'-'*5}  {'-'*7}  {'-'*37}")
        for row in selected:
            c = row["best_cell"]
            print(
                f"  {row['rank']:>4}  {row['node']:<26}{row['family']:<13}"
                f"{row['bin_number']:>5}  {row['score_pp']:>6.1f}pp  "
                f"+={c['positive_shift']:+.1f}pp -={c['negative_shift']:+.1f}pp n={c['bin_n']}"
            )

    print(f"\n  wrote {copied_npz} .npz artifacts under {ws.dir.relative_to(ws.root_dir)}/03_selection/")
    print(f"  wrote {copied_xlsx} workbooks under {ws.dir.relative_to(ws.root_dir)}/03_selection/")
