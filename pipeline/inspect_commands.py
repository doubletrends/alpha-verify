"""Inspection commands for pipeline artifacts."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from engine import barrier, skew as skw, validate as val
from pipeline.runtime import baseline_surface
from tree.tree import all_nodes, find_node, load_tree
from workspace import Workspace


def cmd_status(ws: Workspace) -> None:
    tree = load_tree(ws.tree_path)
    ev = ws.read_json(ws.eval_path).get("nodes", {})
    cl = ws.read_json(ws.cleared_path)
    cleared = {c["node"] for c in cl.get("cleared", [])}
    disc = {t["node"] for t in cl.get("tests", []) if t.get("verdict") == "discovery"}
    sk_disc = set(cl.get("skew_summary", {}).get("discoveries", []))

    cols = [
        "nodes", "cube", "sheet", "sum", "sheet", "skew",
        "valid", "v-sheet", "econ", "sk-disc", "cleared",
    ]
    by_fam = defaultdict(lambda: [0] * len(cols))
    for n in all_nodes(tree):
        c = by_fam[n["family"]]
        c[0] += 1
        c[1] += bool(ws.has_cube(n["family"], n["id"]))
        c[2] += bool(ws.has_surface(n["family"], n["id"]))
        c[3] += bool(ws.has_summary_cube(n["family"], n["id"]))
        c[4] += bool(ws.has_summary_surface(n["family"], n["id"]))
        c[5] += bool(ws.has_skew_sheet(n["family"], n["id"]))
        c[6] += bool(ws.has_validation(n["family"], n["id"]))
        c[7] += bool(ws.has_validation_sheet(n["family"], n["id"]))
        c[8] += bool(ev.get(n["id"], {}).get("passed"))
        c[9] += n["id"] in sk_disc
        c[10] += n["id"] in cleared

    print(f"\n=== Status [{ws.dir.name}] ===")
    print(
        f"  θ {ws.thetas[0]:+.0%}..{ws.thetas[-1]:+.0%}   "
        f"horizons +{ws.horizons[0]}{ws.horizon_unit}..+{ws.horizons[-1]}{ws.horizon_unit}   "
        f"{ws.n_bins} bins"
    )
    if cl:
        meth = cl.get("method", {})
        print(
            f"  gate: BH q={meth.get('q')} over {meth.get('n_tests')} tests, "
            f"{len(disc)} nodes with a discovery, {len(sk_disc)} with a skew discovery"
        )
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
        f"summary {len(ws.summary_thetas)}θ x {len(ws.summary_horizons)}h "
        "(judged and validated here)"
    )


def cmd_read(ws: Workspace, node_id: str) -> None:
    """
    Print one node's summary cube in the terminal: the θ rows carrying a qualifying
    cell, for whichever bin is strongest, with the null test beside them.
    """
    tree = load_tree(ws.tree_path)
    node = find_node(tree, node_id)
    path = ws.summary_cube_path(node["family"], node_id)
    if not path.exists():
        print(f"No summary array for {node_id} - run --summary first.")
        return

    cube = barrier.load_cube(path)
    prob = cube["prob"]
    thetas = cube["thetas"]
    hz = cube["horizons"]
    labels = cube["meta"]["bin_labels"]

    print()
    print(f"{node_id}  [{node['family']}]  {node['feature']}  params={node['params']}")

    baseline = baseline_surface(ws)
    if baseline is None:
        print("  no baseline summary array - run --summary first")
        return
    ev = barrier.evaluate(cube, baseline, ws.min_dev, ws.min_bin_n, ws.min_run)
    best = ev["best"]
    if not best:
        print(f"  no cell reaches {ws.min_dev}pp across {ws.min_run} adjacent θ rows")
        return

    b = best["bin"]
    print(f"  strongest bin : {b + 1} of {len(labels)}   ({labels[b]})")
    print(
        f"  strongest cell: P={best['prob']:.1%} vs {best['base']:.1%} unconditional "
        f"at θ={best['theta']:+.0%}, +{best['horizon']}{ws.horizon_unit} "
        f"({best['dev']:+.1f}pp, run={best['run']}, n={best['bin_n']})"
    )

    vpath = ws.validation_path(node["family"], node_id)
    if vpath.exists():
        vr = val.load(vpath)
        j = int(np.flatnonzero(vr["horizons"] == best["horizon"])[0])
        floor = 1.0 / (1.0 + vr["n_shifts"][j])
        cp = vr["cell_p"][:, b, j]
        tag = " (at the floor)" if vr["peak_p"][j] <= floor + 1e-12 else ""
        print(
            f"  null          : peak {vr['peak_real'][j]:.1f} vs p95 {vr['peak_p95'][j]:.1f}, "
            f"p={vr['peak_p'][j]:.5f}{tag}   floor {floor:.2e}"
        )
        print(
            f"                  {int(np.nansum(cp <= 0.05))} of {int(np.isfinite(cp).sum())} "
            "cells in this bin are pointwise p<=0.05"
        )
        if "skew_peak_p" in vr and np.isfinite(vr["skew_peak_p"]).any():
            k = int(np.nanargmin(np.where(np.isfinite(vr["skew_peak_p"]), vr["skew_peak_p"], np.nan)))
            print(
                f"  skew null     : peak excess {vr['skew_peak_real'][k]:+.1f}pp vs p95 "
                f"{vr['skew_peak_p95'][k]:.1f}, p={vr['skew_peak_p'][k]:.5f} "
                f"at +{int(vr['horizons'][k])}{ws.horizon_unit}"
            )
    else:
        print("  null          : not validated yet")

    spath = ws.skew_path(node["family"], node_id)
    if spath.exists():
        sr = skw.load(spath)
        peaks = skw.peak_by_horizon(sr)
        hit = {h: p for h, p in peaks.items() if p}
        if hit:
            h = max(hit, key=lambda k: abs(hit[k]["excess"]))
            p = hit[h]
            print(
                f"  skew          : excess {p['excess']:+.1f}pp (cond {p['cond']:+.1f}pp) "
                f"at |θ|={p['mag'] * 100:.3g}%, bin {p['bin'] + 1}, +{h}{ws.horizon_unit}"
            )

    avail = {int(x) for x in hz}
    show = [h for h in (1, 2, 3, 5, 7, 10, 14, 21, 30) if h in avail]
    cols = [int(np.flatnonzero(hz == h)[0]) for h in show]
    dev = (prob - baseline[:, None, :]) * 100.0
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
        cells = "".join("       -" if not np.isfinite(v) else f"{v:>8.1%}" for v in prob[i, b, cols])
        print(f"  {format(thetas[i], theta_fmt):>7}{cells}")
        shown += 1
    if not shown:
        print(f"  no θ row deviates by {ws.min_dev}pp at these horizons")
    print()
    print(
        "  values are P(touch θ | condition); rows shown are those deviating "
        f"at least {ws.min_dev}pp"
    )
    print(f"  the workbook carries all {len(thetas)} θ levels, {len(hz)} horizons and {len(labels)} bins")
