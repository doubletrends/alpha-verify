"""Stages 3-5: skew artifacts, validation nulls, and BH gate."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from engine import barrier, skew as skw, validate as val, writer
from pipeline.runtime import baseline_surface, loader, node_feature
from tree.tree import all_in_family, all_nodes, load_tree
from workspace import Workspace

ROOT = Path(__file__).resolve().parents[1]


def cmd_skew(ws: Workspace, family: str | None = None, rerun: bool = False) -> None:
    """Stage 3: compute the up/down asymmetry artifacts from summary cubes."""
    tree = load_tree(ws.tree_path)
    nodes = [
        n for n in (all_in_family(tree, family) if family else all_nodes(tree))
        if ws.has_summary_cube(n["family"], n["id"])
    ]
    if not rerun:
        nodes = [
            n for n in nodes
            if not (ws.has_skew(n["family"], n["id"]) and ws.has_skew_sheet(n["family"], n["id"]))
        ]
    if not nodes:
        print("Nothing to skew (use --rerun to redo, or --summary first).")
        return

    baseline = baseline_surface(ws)
    if baseline is None:
        print("  no baseline summary array - run --summary first")
        return

    th, hz = ws.summary_thetas, ws.summary_horizons
    n_mag = int((len(th) - 1) // 2)
    print(f"\n=== 3. 03_skew_array [{ws.dir.name}] - {len(nodes)} nodes ===")
    print(f"  {n_mag} θ magnitudes x {ws.n_bins} bins x {len(hz)} horizons per node")
    print("  conditional  P(+θ|bin) − P(−θ|bin)     excess  that − baseline skew\n")

    done_npz = 0
    done_xlsx = 0
    for node in nodes:
        try:
            cube = barrier.load_cube(ws.summary_cube_path(node["family"], node["id"]))
            r = skw.skew_from_cube(cube, baseline)
            skw.save(r, ws.skew_path(node["family"], node["id"]), {
                "node": node["id"],
                "family": node["family"],
                "feature": node["feature"],
                "params": node["params"],
                "workspace": ws.dir.name,
                "bin_labels": cube["meta"]["bin_labels"],
                "generated": datetime.now(timezone.utc).isoformat(),
            })
            done_npz += 1
            writer.write_skew_xlsx(
                r,
                ws.skew_sheet_path(node["family"], node["id"]),
                node["id"],
                node["feature"],
                node["params"],
                ws.horizon_unit,
            )
            done_xlsx += 1
        except PermissionError:
            print(f"  {node['id']:<26} [locked] close it in Excel and re-run")
            continue
        except Exception as e:
            print(f"  {node['id']:<26} [skip] {e}")
            continue

        peaks = skw.peak_by_horizon(r)
        hit = {h: p for h, p in peaks.items() if p}
        if hit:
            h = max(hit, key=lambda k: abs(hit[k]["excess"]))
            p = hit[h]
            print(
                f"  {node['id']:<26} peak excess {p['excess']:+5.1f}pp at +{h}"
                f"{ws.horizon_unit}  |θ|={p['mag'] * 100:.3g}%  bin {p['bin'] + 1}  "
                f"(cond {p['cond']:+.1f}pp)"
            )
        else:
            print(f"  {node['id']:<26} no bin above the reporting threshold")

    print(f"\n  wrote {done_npz} .npz artifacts under {ws.dir.relative_to(ROOT)}/03_skew_array/")
    print(f"  wrote {done_xlsx} workbooks under {ws.dir.relative_to(ROOT)}/03_skew_xlsx/")


def cmd_validation(ws: Workspace, family: str | None = None, rerun: bool = False) -> None:
    """Stage 4: null-test the summary cube and write validation artifacts."""
    tree = load_tree(ws.tree_path)
    nodes = [
        n for n in (all_in_family(tree, family) if family else all_nodes(tree))
        if ws.has_summary_cube(n["family"], n["id"])
    ]
    missing_skew = [n["id"] for n in nodes if not ws.has_skew(n["family"], n["id"])]
    nodes = [n for n in nodes if ws.has_skew(n["family"], n["id"])]
    if missing_skew:
        print(
            f"  {len(missing_skew)} node(s) have no skew artifact - run --skew first "
            f"(skipping {', '.join(missing_skew[:5])}{' ...' if len(missing_skew) > 5 else ''})"
        )
    if not rerun:
        nodes = [
            n for n in nodes
            if not ws.has_validation(n["family"], n["id"])
            or not ws.has_validation_sheet(n["family"], n["id"])
        ]
    if not nodes:
        print("Nothing to validate or render (use --rerun to redo, or --skew first).")
        return

    baseline_skew = skw.baseline_cond_skew(barrier.load_cube(ws.baseline_cube))

    get_data = loader(ws)
    th, hz = ws.summary_thetas, ws.summary_horizons
    print(f"\n=== 4. 04_validation_array [{ws.dir.name}] - {len(nodes)} nodes ===")
    print(
        f"  on the summary grid: {len(th)} θ x {ws.n_bins} bins x {len(hz)} horizons "
        f"= {len(th) * ws.n_bins * len(hz)} cells per node"
    )
    print(
        f"  a peak over that has a far lower noise ceiling than one over the full "
        f"{len(ws.thetas) * ws.n_bins * len(ws.horizons)}"
    )
    print(f"  guard {val.EDGE_GUARD} bars either side, so the p-value floor is 1/(usable+1)")
    print("  nulls the touch peak and the excess-skew peak in the same pass\n")

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
                    baseline_skew=baseline_skew,
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

        sk_finite = np.isfinite(r["skew_peak_p"]) if "skew_peak_p" in r else np.array([False])
        if sk_finite.any():
            k = int(np.nanargmin(np.where(sk_finite, r["skew_peak_p"], np.nan)))
            print(
                f"  {'':<26}   skew p={r['skew_peak_p'][k]:.5f} at +{r['horizons'][k]}"
                f"{ws.horizon_unit}  peak {r['skew_peak_real'][k]:5.1f} vs p95 "
                f"{r['skew_peak_p95'][k]:5.1f}"
            )

    print(f"\n  wrote {done_npz} .npz artifacts under {ws.dir.relative_to(ROOT)}/04_validation_array/")
    print(f"  wrote {done_xlsx} workbooks under {ws.dir.relative_to(ROOT)}/04_validation_xlsx/")
    print("  verdicts are assigned by --gate, which needs the whole sweep to correct across")


def cmd_gate(ws: Workspace, q: float = 0.05) -> None:
    """Stage 5: correct across the sweep, then intersect with the economic filter."""
    ev = ws.read_json(ws.eval_path)
    if not ev:
        print("No evaluation.json - run --summary first.")
        return

    tree = load_tree(ws.tree_path)
    rows = []
    skew_rows = []
    for node in all_nodes(tree):
        path = ws.validation_path(node["family"], node["id"])
        if not path.exists():
            continue
        r = val.load(path)
        has_skew = "skew_peak_p" in r
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
            if has_skew and np.isfinite(r["skew_peak_p"][j]):
                skew_rows.append({
                    "node": node["id"],
                    "family": node["family"],
                    "horizon": int(h),
                    "peak_p": float(r["skew_peak_p"][j]),
                    "peak_real": float(r["skew_peak_real"][j]),
                    "peak_p95": float(r["skew_peak_p95"][j]),
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

    skew_summary = {"discovery": 0, "nominal": 0, "n_tests": 0, "discoveries": []}
    if skew_rows:
        sk_p = np.array([r["peak_p"] for r in skew_rows])
        sk_rej, sk_qv = val.bh(sk_p, q)
        for r, rej, qv in zip(skew_rows, sk_rej, sk_qv):
            floor = 1.0 / (1.0 + r["n_shifts"]) if r["n_shifts"] else np.nan
            r["q_value"] = None if not np.isfinite(qv) else round(float(qv), 6)
            r["at_floor"] = bool(
                np.isfinite(r["peak_p"]) and np.isfinite(floor) and r["peak_p"] <= floor + 1e-12
            )
            r["verdict"] = val.verdict(bool(rej), r["peak_p"], r["at_floor"])
        sk_m = int(np.isfinite(sk_p).sum())
        sk_disc = sorted({r["node"] for r in skew_rows if r["verdict"] == "discovery"})
        skew_summary = {
            "discovery": sum(1 for r in skew_rows if r["verdict"] == "discovery"),
            "nominal": sum(1 for r in skew_rows if r["verdict"] == "nominal"),
            "n_tests": sk_m,
            "discoveries": sk_disc,
        }
        print("\n  ── skew (separate BH family, reported not gated) ──")
        print(f"  BH skew discoveries : {skew_summary['discovery']:>4} / {sk_m}   over {len(sk_disc)} node(s)")
        print(f"  nominal (p<=0.05)   : {skew_summary['nominal']:>4} / {sk_m}")
        if sk_disc:
            print(f"  {', '.join(sk_disc)}")
    else:
        print("\n  ── skew ── no skew p-values in the validation artifacts (re-run --skew then --validation --rerun)")

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
        "skew_summary": skew_summary,
        "skew_tests": skew_rows,
    })
    print(f"\n  wrote {ws.cleared_path.relative_to(ROOT)}")
