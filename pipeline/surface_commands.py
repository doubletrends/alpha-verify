"""Stages 1 and 2: surface measurement, summary selection, and workbook rendering."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from engine import barrier, writer
from pipeline.runtime import baseline_surface, loader, node_feature
from tree.tree import all_in_family, all_nodes, load_tree
from workspace import BASELINE_NODE, Workspace

ROOT = Path(__file__).resolve().parents[1]


def build_cube(ws: Workspace, node: dict, get_data, quiet: bool = False) -> None:
    """Measure one node across every horizon and barrier level; write the full cube."""
    data, feat = node_feature(ws, node, get_data)
    edges = barrier.bin_edges(feat, ws.n_bins)
    cube = barrier.touch_tensor(data, feat, ws.horizons, ws.thetas, edges)

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
    tree = load_tree(ws.tree_path)
    nodes = all_in_family(tree, family) if family else all_nodes(tree)
    if not rerun:
        nodes = [n for n in nodes if not ws.has_cube(n["family"], n["id"])]
    if not nodes:
        print("No missing surface arrays (use --rerun to rebuild existing arrays).")
        return

    get_data = loader(ws)
    hz = ws.horizons
    print(f"\n=== 1. 01_surface_array [{ws.dir.name}] - {len(nodes)} nodes ===")
    print(
        f"  {len(ws.thetas)} θ x {ws.n_bins} bins x {len(hz)} horizons "
        f"(+{hz[0]}{ws.horizon_unit}..+{hz[-1]}{ws.horizon_unit})   "
        f"θ {ws.thetas[0]:+.0%}..{ws.thetas[-1]:+.0%} step {ws.theta_step:.0%}"
    )
    print("  value = P(touch θ in h | bin); intraday high/low\n")

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
    _render(ws, family, summary=False)


def cmd_surface(ws: Workspace, family: str | None = None, rerun: bool = False) -> None:
    """Stage 1. Write the full surface array and workbook."""
    _write_surface_array(ws, family, rerun)
    _write_surface_xlsx(ws, family)


def _write_summary_array(ws: Workspace, family: str | None = None) -> None:
    """
    Stage 2. The coarse grid everything is judged on, selected from the full cube.

    Pure index selection means every summary cell is bit-identical to the full cell it
    came from. The economic filter runs here so judged and validated grids match.
    """
    tree = load_tree(ws.tree_path)
    nodes = [
        n for n in (all_in_family(tree, family) if family else all_nodes(tree))
        if ws.has_cube(n["family"], n["id"])
    ]
    if not nodes:
        print("No full surface arrays to summarize - run --surface first.")
        return

    nodes = (
        [n for n in nodes if n["id"] == BASELINE_NODE]
        + [n for n in nodes if n["id"] != BASELINE_NODE]
    )

    th, hz = ws.summary_thetas, ws.summary_horizons
    print(f"\n=== 2. 02_summary_array [{ws.dir.name}] - {len(nodes)} nodes ===")
    print(
        f"  {len(th)} θ x {ws.n_bins} bins x {len(hz)} horizons = "
        f"{len(th) * ws.n_bins * len(hz)} cells  "
        f"(full: {len(ws.thetas) * ws.n_bins * len(ws.horizons)})"
    )
    print(f"  θ {', '.join(f'{v:+.0%}' for v in th[len(th)//2:])}  mirrored")
    print(f"  h {', '.join(f'+{h}{ws.horizon_unit}' for h in hz)}\n")

    evals, skipped = {}, {}
    for node in nodes:
        try:
            full = barrier.load_cube(ws.cube_path(node["family"], node["id"]))
            sub = barrier.summarize(full, th, hz)
            barrier.save_cube(
                sub,
                ws.summary_cube_path(node["family"], node["id"]),
                {**full["meta"], "grid": "summary"},
            )
        except Exception as e:
            skipped[node["id"]] = str(e)
            print(f"  {node['id']:<26} [skip] {e}")
            continue

        baseline = baseline_surface(ws)
        if baseline is None:
            print("  no baseline summary yet - cannot evaluate")
            return
        ev = barrier.evaluate(sub, baseline, ws.min_dev, ws.min_bin_n, ws.min_run)
        evals[node["id"]] = ev
        b = ev["best"]
        note = (
            f"best {b['dev']:+5.1f}pp  θ={b['theta']:+.0%}  "
            f"+{b['horizon']}{ws.horizon_unit:<3} bin {b['bin']}  n={b['bin_n']}  "
            f"run={b['run']}"
        ) if b else "no qualifying cell"
        print(f"  {node['id']:<26} {note}")

    ws.write_json(ws.eval_path, {
        "workspace": ws.dir.name,
        "generated": datetime.now(timezone.utc).isoformat(),
        "grid": {"thetas": th.tolist(), "horizons": hz.tolist()},
        "criteria": {"min_dev": ws.min_dev, "min_bin_n": ws.min_bin_n, "min_run": ws.min_run},
        "passed": sorted(n for n, e in evals.items() if e["passed"]),
        "nodes": {n: {"passed": e["passed"], "best": e["best"]} for n, e in evals.items()},
        "skipped": skipped,
    })
    n_pass = sum(1 for e in evals.values() if e["passed"])
    print(
        f"\n  economic filter: {n_pass} / {len(evals)} pass "
        f"(|dev| >= {ws.min_dev}pp across >= {ws.min_run} adjacent θ rows, "
        f"bin n >= {ws.min_bin_n})"
    )
    print(f"  wrote {ws.eval_path.relative_to(ROOT)}")


def _write_summary_xlsx(ws: Workspace, family: str | None = None) -> None:
    """Stage 2. The summary cube, made readable. Same renderer, smaller grid."""
    _render(ws, family, summary=True)


def cmd_summary(ws: Workspace, family: str | None = None) -> None:
    """Stage 2. Write the summary array and workbook."""
    _write_summary_array(ws, family)
    _write_summary_xlsx(ws, family)


def _render(ws: Workspace, family: str | None, summary: bool) -> None:
    tree = load_tree(ws.tree_path)
    have = ws.has_summary_cube if summary else ws.has_cube
    src = ws.summary_cube_path if summary else ws.cube_path
    dst = ws.summary_surface_path if summary else ws.surface_path
    label = "summary" if summary else "full"
    stage = "2. 02_summary_xlsx" if summary else "1. 01_surface_xlsx"
    out_dir = "02_summary_xlsx" if summary else "01_surface_xlsx"

    nodes = [
        n for n in (all_in_family(tree, family) if family else all_nodes(tree))
        if have(n["family"], n["id"])
    ]
    if not nodes:
        print(f'No {label} arrays to render - run --{"summary" if summary else "surface"} first.')
        return

    print(f"\n=== {stage} [{ws.dir.name}] - {len(nodes)} nodes ===")
    print(
        f"  {ws.n_bins} tabs per node, one per condition bin; "
        "each tab is that bin's full θ x horizon face\n"
    )
    n_ok = 0
    for node in nodes:
        cube = barrier.load_cube(src(node["family"], node["id"]))
        try:
            writer.write_barrier_xlsx(
                cube,
                dst(node["family"], node["id"]),
                node["id"],
                node["feature"],
                node["params"],
                ws.horizon_unit,
            )
        except PermissionError:
            print(f"  {node['id']:<26} [locked] close it in Excel and re-run")
            continue
        n_ok += 1
    print(f"  wrote {n_ok} workbooks under {ws.dir.relative_to(ROOT)}/{out_dir}/")
