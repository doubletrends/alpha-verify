"""Stage 2: baseline-subtracted shift artifacts and workbooks."""

from __future__ import annotations

from domain import barrier, shift
from infrastructure.workspace import BASELINE_NODE, Workspace
from pipeline.context import baseline_surface
from presentation import workbooks


def _write_shift_array(ws: Workspace) -> None:
    nodes = [
        node for node in ws.catalog.all_nodes()
        if ws.has_cube(node["family"], node["id"])
    ]
    if not nodes:
        print("No full surface arrays to shift - run surface first.")
        return

    baseline = baseline_surface(ws)
    if baseline is None:
        print("No baseline surface array - run surface first.")
        return

    nodes = [node for node in nodes if node["id"] == BASELINE_NODE] + [
        node for node in nodes if node["id"] != BASELINE_NODE
    ]
    deltas, horizons = ws.deltas, ws.horizons
    print(f"\n=== 2. 02_shift arrays [{ws.dir.name}] - {len(nodes)} nodes ===")
    print(
        f"  {len(deltas)} Delta x {ws.n_bins} bins x {len(horizons)} horizons = "
        f"{len(deltas) * ws.n_bins * len(horizons)} cells"
    )
    print("  value = P(touch Delta in t | bin) - P(touch Delta in t baseline), percentage points")
    print("  red = more frequent than baseline; blue = less frequent\n")

    skipped = {}
    for node in nodes:
        try:
            full = barrier.load_cube(ws.cube_path(node["family"], node["id"]))
            shifted = shift.from_cube(full, baseline)
            shift.save(
                shifted,
                ws.shift_cube_path(node["family"], node["id"]),
                {**full["meta"], "grid": "shift", "value": "conditional_minus_baseline_pp"},
            )
        except Exception as error:
            skipped[node["id"]] = str(error)
            print(f"  {node['id']:<26} [skip] {error}")
            continue
        print(f"  {node['id']:<26} {shifted['shift'].shape}")
    if skipped:
        print(f"\n  skipped {len(skipped)} nodes")


def _render_shift(ws: Workspace) -> None:
    nodes = [
        node for node in ws.catalog.all_nodes()
        if ws.has_shift_cube(node["family"], node["id"])
    ]
    if not nodes:
        print("No shift arrays to render - run shift first.")
        return

    print(f"\n=== 2. 02_shift workbooks [{ws.dir.name}] - {len(nodes)} nodes ===")
    print(
        f"  {ws.n_bins} tabs per node, one per condition bin; "
        "red/blue cells show baseline-subtracted touch probability\n"
    )
    written = 0
    for node in nodes:
        cube = shift.load(ws.shift_cube_path(node["family"], node["id"]))
        try:
            workbooks.write_shift_xlsx(
                cube,
                ws.shift_surface_path(node["family"], node["id"]),
                node["id"],
                node["feature"],
                node["params"],
                ws.horizon_unit,
            )
        except PermissionError:
            print(f"  {node['id']:<26} [locked] close it in Excel and re-run")
            continue
        written += 1
    print(f"  wrote {written} workbooks under {ws.dir.relative_to(ws.root_dir)}/02_shift/")


def cmd_shift(ws: Workspace) -> None:
    """Write full baseline-subtracted shift arrays and workbooks."""
    _write_shift_array(ws)
    _render_shift(ws)
