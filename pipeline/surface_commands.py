"""Stages 1 and 2: surface measurement, shift artifacts, and workbook rendering."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from engine import barrier, shift, writer
from pipeline.runtime import baseline_surface, loader, node_feature
from universe import all_in_family, all_nodes, load_universe
from workspace import BASELINE_NODE, Workspace

ROOT = Path(__file__).resolve().parents[1]


def build_cube(ws: Workspace, node: dict, get_data, quiet: bool = False) -> None:
    """Measure one node across every horizon and barrier level; write the full cube."""
    data, feat = node_feature(ws, node, get_data)
    edges = barrier.bin_edges(feat, ws.n_bins)
    cube = barrier.touch_tensor(data, feat, ws.horizons, ws.thetas, edges)
    cube["index"] = data.index.astype(str).to_numpy()
    cube["feature_values"] = feat.reindex(data.index).to_numpy(float)
    for col in ("open", "high", "low", "close", "volume"):
        if col in data:
            cube[col] = data[col].to_numpy(float)

    barrier.save_cube(cube, ws.cube_path(node["family"], node["id"]), {
        "node": node["id"],
        "family": node["family"],
        "feature": node["feature"],
        "params": node["params"],
        "workspace": ws.dir.name,
        "bin_labels": barrier.bin_labels(edges, feat),
        "grid": "full",
        "generated": datetime.now(timezone.utc).isoformat(),
    })
    if not quiet:
        print(f"  {node['id']:<26} {cube['prob'].shape}  n={int(cube['n_obs'][0])}")


def _write_surface_array(ws: Workspace, family: str | None = None, rerun: bool = False) -> None:
    """Stage 1a. The faithful record: every barrier level, every horizon."""
    universe = load_universe(ws.universe_path)
    nodes = all_in_family(universe, family) if family else all_nodes(universe)
    if not rerun:
        nodes = [n for n in nodes if not ws.has_cube(n["family"], n["id"])]
    if not nodes:
        print("No missing surface arrays (use --rerun to rebuild existing arrays).")
        return

    get_data = loader(ws)
    hz = ws.horizons
    print(f"\n=== 1. 01_surface_array [{ws.dir.name}] - {len(nodes)} nodes ===")
    print(
        f"  {len(ws.thetas)} theta x {ws.n_bins} bins x {len(hz)} horizons "
        f"(+{hz[0]}{ws.horizon_unit}..+{hz[-1]}{ws.horizon_unit})   "
        f"theta {ws.thetas[0]:+.0%}..{ws.thetas[-1]:+.0%} step {ws.theta_step:.0%}"
    )
    print("  value = P(touch theta in h | bin); intraday high/low\n")

    skipped = {}
    for node in nodes:
        try:
            build_cube(ws, node, get_data)
        except Exception as e:
            skipped[node["id"]] = str(e)
            print(f"  {node['id']:<26} [skip] {e}")
    print(
        f"\n  wrote to {ws.dir.relative_to(ROOT)}/01_surface_array/"
        + (f"   ({len(skipped)} skipped)" if skipped else "")
    )


def _write_surface_xlsx(ws: Workspace, family: str | None = None) -> None:
    """Stage 1. The full cube, made readable; never re-measures."""
    _render_surface(ws, family)


def cmd_surface(ws: Workspace, family: str | None = None, rerun: bool = False) -> None:
    """Stage 1. Write the full surface array and workbook."""
    _write_surface_array(ws, family, rerun)
    _write_surface_xlsx(ws, family)


def _write_shift_array(ws: Workspace, family: str | None = None) -> None:
    """
    Stage 2. Full conditional surfaces after subtracting the baseline surface.

    This is the grid everything downstream reads and judges. It has the same theta and
    horizon axes as Stage 1; the value is the deviation from the baseline in percentage
    points.
    """
    universe = load_universe(ws.universe_path)
    nodes = [
        n for n in (all_in_family(universe, family) if family else all_nodes(universe))
        if ws.has_cube(n["family"], n["id"])
    ]
    if not nodes:
        print("No full surface arrays to shift - run --surface first.")
        return

    baseline = baseline_surface(ws)
    if baseline is None:
        print("No baseline surface array - run --surface first.")
        return

    nodes = (
        [n for n in nodes if n["id"] == BASELINE_NODE]
        + [n for n in nodes if n["id"] != BASELINE_NODE]
    )

    th, hz = ws.shift_thetas, ws.shift_horizons
    print(f"\n=== 2. 02_shift_array [{ws.dir.name}] - {len(nodes)} nodes ===")
    print(f"  {len(th)} theta x {ws.n_bins} bins x {len(hz)} horizons = {len(th) * ws.n_bins * len(hz)} cells")
    print("  value = P(touch theta in h | bin) - P(touch theta in h baseline), percentage points")
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
        except Exception as e:
            skipped[node["id"]] = str(e)
            print(f"  {node['id']:<26} [skip] {e}")
            continue

        print(f"  {node['id']:<26} {shifted['shift'].shape}")
    if skipped:
        print(f"\n  skipped {len(skipped)} nodes")


def _write_shift_xlsx(ws: Workspace, family: str | None = None) -> None:
    """Stage 2. The full shift cube, made readable."""
    _render_shift(ws, family)


def cmd_shift(ws: Workspace, family: str | None = None) -> None:
    """Stage 2. Write the full baseline-subtracted shift array and workbook."""
    _write_shift_array(ws, family)
    _write_shift_xlsx(ws, family)


def _render_surface(ws: Workspace, family: str | None) -> None:
    universe = load_universe(ws.universe_path)
    nodes = [
        n for n in (all_in_family(universe, family) if family else all_nodes(universe))
        if ws.has_cube(n["family"], n["id"])
    ]
    if not nodes:
        print("No full arrays to render - run --surface first.")
        return

    print(f"\n=== 1. 01_surface_xlsx [{ws.dir.name}] - {len(nodes)} nodes ===")
    print(
        f"  {ws.n_bins} tabs per node, one per condition bin; "
        "each tab is that bin's full theta x horizon face\n"
    )
    n_ok = 0
    for node in nodes:
        cube = barrier.load_cube(ws.cube_path(node["family"], node["id"]))
        try:
            writer.write_barrier_xlsx(
                cube,
                ws.surface_path(node["family"], node["id"]),
                node["id"],
                node["feature"],
                node["params"],
                ws.horizon_unit,
            )
        except PermissionError:
            print(f"  {node['id']:<26} [locked] close it in Excel and re-run")
            continue
        n_ok += 1
    print(f"  wrote {n_ok} workbooks under {ws.dir.relative_to(ROOT)}/01_surface_xlsx/")


def _render_shift(ws: Workspace, family: str | None) -> None:
    universe = load_universe(ws.universe_path)
    nodes = [
        n for n in (all_in_family(universe, family) if family else all_nodes(universe))
        if ws.has_shift_cube(n["family"], n["id"])
    ]
    if not nodes:
        print("No shift arrays to render - run --shift first.")
        return

    print(f"\n=== 2. 02_shift_xlsx [{ws.dir.name}] - {len(nodes)} nodes ===")
    print(
        f"  {ws.n_bins} tabs per node, one per condition bin; "
        "red/blue cells show baseline-subtracted touch probability\n"
    )
    n_ok = 0
    for node in nodes:
        cube = shift.load(ws.shift_cube_path(node["family"], node["id"]))
        try:
            writer.write_shift_xlsx(
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
        n_ok += 1
    print(f"  wrote {n_ok} workbooks under {ws.dir.relative_to(ROOT)}/02_shift_xlsx/")
