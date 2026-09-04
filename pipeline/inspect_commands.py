"""Inspection commands for pipeline artifacts."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from engine import selection, shift, validate as val
from pipeline.stat_commands import _validation_matches_shift
from universe import all_nodes, find_node, load_universe
from workspace import Workspace


def _gate_is_current(ws: Workspace, universe: dict, cleared: dict) -> bool:
    if not cleared:
        return False
    sel = ws.read_json(ws.selection_path)
    selected = [r for r in sel.get("selected", []) if ws.has_selection_array(r)]
    if not selected:
        return False
    valid_sheets = set()
    by_id = {n["id"]: n for n in all_nodes(universe)}
    for row in selected:
        node = by_id.get(row["node"])
        if node is None:
            continue
        if not _validation_matches_shift(ws, node):
            continue
        valid_sheets.add((row["node"], int(row["bin"])))
    tests = cleared.get("tests", [])
    if not valid_sheets:
        return False
    expected = len(valid_sheets) * len(ws.shift_horizons)
    if len(tests) != expected:
        return False
    return all((t.get("node"), int(t.get("bin", -1))) in valid_sheets for t in tests)


def cmd_status(ws: Workspace) -> None:
    universe = load_universe(ws.universe_path)
    sel = ws.read_json(ws.selection_path)
    selected = [r for r in sel.get("selected", []) if ws.has_selection_array(r)]
    selected_bins = selection.selected_bins_by_node({"selected": selected})
    cl = ws.read_json(ws.cleared_path)
    gate_current = _gate_is_current(ws, universe, cl)
    cleared_rows = cl.get("cleared", []) if gate_current else []
    cleared = {c["node"] for c in cleared_rows}
    disc = {t["node"] for t in cl.get("tests", []) if t.get("verdict") == "discovery"} if gate_current else set()

    cols = [
        "nodes", "cube", "sheet", "shift", "sheet", "select", "s-sheet", "valid", "v-sheet", "econ", "cleared",
    ]
    by_fam = defaultdict(lambda: [0] * len(cols))
    for n in all_nodes(universe):
        c = by_fam[n["family"]]
        c[0] += 1
        c[1] += bool(ws.has_cube(n["family"], n["id"]))
        c[2] += bool(ws.has_surface(n["family"], n["id"]))
        c[3] += bool(ws.has_shift_cube(n["family"], n["id"]))
        c[4] += bool(ws.has_shift_surface(n["family"], n["id"]))
        selected_for_node = selected_bins.get(n["id"], set())
        selected_rows = [r for r in selected if r["node"] == n["id"]]
        c[5] += sum(1 for r in selected_rows if ws.has_selection_array(r))
        c[6] += sum(1 for r in selected_rows if ws.has_selection_surface(r))
        valid_current = _validation_matches_shift(ws, n)
        c[7] += len(selected_for_node) if valid_current else 0
        c[8] += len(selected_for_node) if valid_current and ws.has_validation_sheet(n["family"], n["id"]) else 0
        c[9] += sum(1 for r in selected_rows if r.get("economic", {}).get("passed"))
        c[10] += sum(1 for r in cleared_rows if r.get("node") == n["id"])

    print(f"\n=== Status [{ws.dir.name}] ===")
    print(
        f"  θ {ws.thetas[0]:+.0%}..{ws.thetas[-1]:+.0%}   "
        f"horizons +{ws.horizons[0]}{ws.horizon_unit}..+{ws.horizons[-1]}{ws.horizon_unit}   "
        f"{ws.n_bins} bins"
    )
    if cl and gate_current:
        meth = cl.get("method", {})
        print(
            f"  gate: BH q={meth.get('q')} over {meth.get('n_tests')} tests, "
            f"{len(disc)} nodes with a discovery, {len(cleared_rows)} cleared selected sheets"
        )
    elif cl:
        print("  gate: stale for current selection - run --validation, then --gate")
    print()
    hdr = f"  {'family':<15}" + "".join(f"{name:>9}" for name in cols)
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for fam in sorted(by_fam):
        print(f"  {fam:<15}" + "".join(f"{v:>9}" for v in by_fam[fam]))
    tot = [sum(by_fam[f][i] for f in by_fam) for i in range(len(cols))]
    print("  " + "-" * (len(hdr) - 2))
    print(f"  {'TOTAL':<15}" + "".join(f"{v:>9}" for v in tot))
    print(
        f"\n  full grid {len(ws.thetas)}θ x {len(ws.horizons)}h   |   "
        f"03_selection keeps top {len(selected)} node/bin sheets"
    )


def cmd_read(ws: Workspace, node_id: str) -> None:
    """
    Print one node's shift cube in the terminal: the θ rows carrying a qualifying
    cell, for whichever bin is strongest, with the null test beside them.
    """
    universe = load_universe(ws.universe_path)
    node = find_node(universe, node_id)
    path = ws.shift_cube_path(node["family"], node_id)
    if not path.exists():
        print(f"No shift array for {node_id} - run --shift first.")
        return

    cube = shift.load(path)
    dev = cube["shift"]
    thetas = cube["thetas"]
    hz = cube["horizons"]
    labels = cube["meta"]["bin_labels"]

    print()
    print(f"{node_id}  [{node['family']}]  {node['feature']}  params={node['params']}")

    ev = shift.evaluate(cube, ws.min_dev, ws.min_bin_n, ws.min_run)
    best = ev["best"]
    if not best:
        print(f"  no cell reaches {ws.min_dev}pp across {ws.min_run} adjacent θ rows")
        return

    b = best["bin"]
    print(f"  strongest bin : {b + 1} of {len(labels)}   ({labels[b]})")
    print(
        f"  strongest cell: shift={best['dev']:+.1f}pp; P={best['prob']:.1%} "
        f"vs {best['base']:.1%} baseline "
        f"at θ={best['theta']:+.0%}, +{best['horizon']}{ws.horizon_unit} "
        f"(run={best['run']}, n={best['bin_n']})"
    )

    vpath = ws.validation_path(node["family"], node_id)
    if _validation_matches_shift(ws, node):
        vr = val.load(vpath)
        j = int(np.flatnonzero(vr["horizons"] == best["horizon"])[0])
        floor = 1.0 / (1.0 + vr["n_shifts"][j])
        cp = vr["cell_p"][:, b, j]
        peak_real = vr["sheet_peak_real"][b, j] if "sheet_peak_real" in vr else vr["peak_real"][j]
        peak_p95 = vr["sheet_peak_p95"][b, j] if "sheet_peak_p95" in vr else vr["peak_p95"][j]
        peak_p = vr["sheet_peak_p"][b, j] if "sheet_peak_p" in vr else vr["peak_p"][j]
        tag = " (at the floor)" if peak_p <= floor + 1e-12 else ""
        print(
            f"  null          : sheet peak {peak_real:.1f} vs p95 {peak_p95:.1f}, "
            f"p={peak_p:.5f}{tag}   floor {floor:.2e}"
        )
        print(
            f"                  {int(np.nansum(cp <= 0.05))} of {int(np.isfinite(cp).sum())} "
            "cells in this bin are pointwise p<=0.05"
        )
    else:
        print("  null          : not validated yet")

    avail = {int(x) for x in hz}
    show = [h for h in (1, 2, 3, 5, 7, 10, 14, 21, 30) if h in avail]
    cols = [int(np.flatnonzero(hz == h)[0]) for h in show]
    pct_dec = 1 if ws.theta_step < 0.01 else 0
    theta_fmt = f"+.{pct_dec}%"

    print()
    header = "".join(f"{'+' + str(h) + ws.horizon_unit:>8}" for h in show)
    print(f"  {'θ':>7}{header}")
    print("  " + "-" * (7 + 8 * len(show)))
    shown = 0
    for i in sorted(range(len(thetas)), key=lambda k: -thetas[k]):
        d = dev[i, b, cols]
        finite = d[np.isfinite(d)]
        if finite.size == 0 or np.max(np.abs(finite)) < ws.min_dev:
            continue
        cells = "".join("       -" if not np.isfinite(v) else f"{v:>8.1f}" for v in dev[i, b, cols])
        print(f"  {format(thetas[i], theta_fmt):>7}{cells}")
        shown += 1
    if not shown:
        print(f"  no θ row deviates by {ws.min_dev}pp at these horizons")
    print()
    print(
        "  values are percentage-point shifts from baseline; rows shown deviate "
        f"at least {ws.min_dev}pp"
    )
    print(f"  the workbook carries all {len(thetas)} θ levels, {len(hz)} horizons and {len(labels)} bins")
