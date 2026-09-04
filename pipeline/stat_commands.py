"""Stages 3-4: validation nulls and BH gate."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from engine import barrier, validate as val, writer
from pipeline.runtime import loader, node_feature
from universe import all_in_family, all_nodes, load_universe
from workspace import Workspace

ROOT = Path(__file__).resolve().parents[1]


def cmd_validation(ws: Workspace, family: str | None = None, rerun: bool = False) -> None:
    """Stage 3: null-test the summary cube and write validation artifacts."""
    universe = load_universe(ws.universe_path)
    nodes = [
        n for n in (all_in_family(universe, family) if family else all_nodes(universe))
        if ws.has_summary_cube(n["family"], n["id"])
    ]
    if not rerun:
        nodes = [
            n for n in nodes
            if not ws.has_validation(n["family"], n["id"])
            or not ws.has_validation_sheet(n["family"], n["id"])
        ]
    if not nodes:
        print("Nothing to validate or render (use --rerun to redo, or --summary first).")
        return

    get_data = loader(ws)
    th, hz = ws.summary_thetas, ws.summary_horizons
    print(f"\n=== 3. 03_validation_array [{ws.dir.name}] - {len(nodes)} nodes ===")
    print(
        f"  on the summary grid: {len(th)} θ x {ws.n_bins} bins x {len(hz)} horizons "
        f"= {len(th) * ws.n_bins * len(hz)} cells per node"
    )
    print(
        f"  a peak over that has a far lower noise ceiling than one over the full "
        f"{len(ws.thetas) * ws.n_bins * len(ws.horizons)}"
    )
    print(f"  guard {val.EDGE_GUARD} bars either side, so the p-value floor is 1/(usable+1)")
    print("  nulls the strongest touch-probability deviation per horizon\n")

    done_npz = 0
    done_xlsx = 0
    for node in nodes:
        try:
            cube = barrier.load_cube(ws.summary_cube_path(node["family"], node["id"]))
            if ws.has_validation(node["family"], node["id"]) and not rerun:
                r = val.load(ws.validation_path(node["family"], node["id"]))
            else:
                data, feat = node_feature(ws, node, get_data)
                r = val.validate_node(
                    data,
                    feat,
                    cube["horizons"],
                    cube["thetas"],
                    cube["edges"],
                )
                val.save(r, ws.validation_path(node["family"], node["id"]), {
                    "node": node["id"],
                    "family": node["family"],
                    "feature": node["feature"],
                    "params": node["params"],
                    "workspace": ws.dir.name,
                    "bin_labels": cube["meta"]["bin_labels"],
                    "generated": datetime.now(timezone.utc).isoformat(),
                })
                done_npz += 1
            writer.write_validation_xlsx(
                r,
                ws.validation_sheet_path(node["family"], node["id"]),
                node["id"],
                node["feature"],
                node["params"],
                ws.horizon_unit,
            )
            done_xlsx += 1
        except Exception as e:
            print(f"  {node['id']:<26} [skip] {e}")
            continue

        finite = np.isfinite(r["peak_p"])
        if finite.any():
            k = int(np.nanargmin(np.where(finite, r["peak_p"], np.nan)))
            floor = 1.0 / (1.0 + r["n_shifts"][k])
            tag = "  [at the floor]" if r["peak_p"][k] <= floor + 1e-12 else ""
            print(
                f"  {node['id']:<26} best p={r['peak_p'][k]:.5f} at +{r['horizons'][k]}"
                f"{ws.horizon_unit}  peak {r['peak_real'][k]:5.1f} vs p95 "
                f"{r['peak_p95'][k]:5.1f}{tag}"
            )
        else:
            print(f"  {node['id']:<26} insufficient data at every horizon")

    print(f"\n  wrote {done_npz} .npz artifacts under {ws.dir.relative_to(ROOT)}/03_validation_array/")
    print(f"  wrote {done_xlsx} workbooks under {ws.dir.relative_to(ROOT)}/03_validation_xlsx/")
    print("  verdicts are assigned by --gate, which needs the whole sweep to correct across")


def cmd_gate(ws: Workspace, q: float = 0.05) -> None:
    """Stage 4: correct across the sweep, then intersect with the economic filter."""
    ev = ws.read_json(ws.eval_path)
    if not ev:
        print("No evaluation.json - run --summary first.")
        return

    universe = load_universe(ws.universe_path)
    rows = []
    for node in all_nodes(universe):
        path = ws.validation_path(node["family"], node["id"])
        if not path.exists():
            continue
        r = val.load(path)
        for j, h in enumerate(r["horizons"]):
            rows.append({
                "node": node["id"],
                "family": node["family"],
                "horizon": int(h),
                "peak_p": float(r["peak_p"][j]),
                "peak_real": float(r["peak_real"][j]),
                "peak_p95": float(r["peak_p95"][j]),
                "n_shifts": int(r["n_shifts"][j]),
            })
    if not rows:
        print("No validation artifacts - run --validation first.")
        return

    pvals = np.array([r["peak_p"] for r in rows])
    rejected, qvals = val.bh(pvals, q)
    for r, rej, qv in zip(rows, rejected, qvals):
        floor = 1.0 / (1.0 + r["n_shifts"]) if r["n_shifts"] else np.nan
        r["q_value"] = None if not np.isfinite(qv) else round(float(qv), 6)
        r["at_floor"] = bool(
            np.isfinite(r["peak_p"]) and np.isfinite(floor) and r["peak_p"] <= floor + 1e-12
        )
        r["verdict"] = val.verdict(bool(rej), r["peak_p"], r["at_floor"])

    m = int(np.isfinite(pvals).sum())
    n_disc = sum(1 for r in rows if r["verdict"] == "discovery")
    n_nom = sum(1 for r in rows if r["verdict"] == "nominal")
    floors = sorted({r["n_shifts"] for r in rows if r["n_shifts"]})
    floor_lo = 1.0 / (1.0 + max(floors)) if floors else float("nan")

    print(f"\n=== Gate [{ws.dir.name}] ===")
    print(f"  {m} tests over {len({r['node'] for r in rows})} nodes x {len({r['horizon'] for r in rows})} horizons")
    print(
        f"  Benjamini-Hochberg at q = {q}   |   p-value floor ~ {floor_lo:.2e} "
        "(exact null has only n distinct shifts)"
    )
    print(
        f"  Bonferroni would need p <= {0.05 / max(m, 1):.2e}, "
        f"{'reachable' if 0.05 / max(m, 1) >= floor_lo else 'BELOW the floor - unusable'}\n"
    )
    print(f"  BH discoveries    : {n_disc:>4} / {m}")
    print(f"  nominal (p<=0.05) : {n_nom:>4} / {m}")

    passed_econ = set(ev.get("passed", []))
    by_node: dict[str, dict] = {}
    for r in rows:
        if r["verdict"] != "discovery":
            continue
        cur = by_node.get(r["node"])
        if cur is None or r["q_value"] < cur["q_value"]:
            by_node[r["node"]] = r

    cleared = []
    for nid, r in by_node.items():
        if nid not in passed_econ:
            continue
        best = ev["nodes"].get(nid, {}).get("best")
        cleared.append({
            "node": nid,
            "family": r["family"],
            "horizon": r["horizon"],
            "q_value": r["q_value"],
            "peak_p": r["peak_p"],
            "peak_real": r["peak_real"],
            "at_floor": r["at_floor"],
            "best_cell": best,
        })
    cleared.sort(key=lambda c: (c["q_value"], -abs(c["best_cell"]["dev"] if c["best_cell"] else 0)))

    print(f"\n  economic filter passed : {len(passed_econ)}")
    print(f"  and a BH discovery     : {len(cleared)}\n")
    if cleared:
        print(f"  {'node':<26}{'family':<13}{'h':>5}  {'q':<10}{'peak':>7}  best cell")
        print(f"  {'-'*26}{'-'*13}{'-'*5}  {'-'*10}{'-'*7}  {'-'*36}")
        for c in cleared:
            b = c["best_cell"] or {}
            cell = (
                f"{b.get('dev', 0):+.1f}pp @ θ={b.get('theta', 0):+.0%} "
                f"bin {b.get('bin')} n={b.get('bin_n')}"
            ) if b else ""
            print(
                f"  {c['node']:<26}{c['family']:<13}{c['horizon']:>5}  "
                f"{c['q_value']:<10.5f}{c['peak_real']:>7.1f}  {cell}"
            )
    else:
        print("  Nothing clears both filters.")

    ws.write_json(ws.cleared_path, {
        "workspace": ws.dir.name,
        "generated": datetime.now(timezone.utc).isoformat(),
        "method": {
            "correction": "benjamini-hochberg",
            "q": q,
            "n_tests": m,
            "p_floor": floor_lo,
            "note": "circular-shift null is exact over all n shifts; the floor 1/(n+1) "
                    "is below no Bonferroni threshold at this m, so FDR is used instead of FWER",
        },
        "summary": {"discovery": n_disc, "nominal": n_nom, "cleared": len(cleared)},
        "cleared": cleared,
        "tests": rows,
    })
    print(f"\n  wrote {ws.cleared_path.relative_to(ROOT)}")
