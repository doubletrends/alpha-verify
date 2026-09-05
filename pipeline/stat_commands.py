"""Stages 4-5: selected-sheet validation nulls and BH gate."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from engine import selection, shift, validate as val, writer
from pipeline.runtime import artifact_node_feature
from workspace import Workspace

ROOT = Path(__file__).resolve().parents[1]


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
    sel = ws.read_json(ws.selection_path)
    rows = sel.get("selected", [])
    if family:
        rows = [r for r in rows if r.get("family") == family]
    return rows


def cmd_validation(ws: Workspace, family: str | None = None, rerun: bool = False) -> None:
    """Stage 4: null-test nodes that own selected bin sheets."""
    rows = _selected_rows(ws, family)
    if not rows:
        print("No selected sheets - run --selection first.")
        return
    rows = [r for r in rows if ws.has_selection_array(r)]
    if not rows:
        print("No 03_selection_array artifacts - run --selection first.")
        return
    by_id = {n["id"]: n for n in ws.catalog.all_nodes()}
    rows = [
        r for r in rows
        if r["node"] in by_id and ws.has_shift_cube(r["family"], r["node"])
    ]
    if not rerun:
        rows = [
            r for r in rows
            if not _validation_matches_selection(ws, r)
            or not ws.has_validation_surface(r)
        ]
    if not rows:
        print("Nothing to validate or render (use --rerun to redo selected sheets).")
        return

    th, ts = ws.shift_deltas, ws.shift_horizons
    print(f"\n=== 4. 04_validation_array [{ws.dir.name}] - {len(rows)} selected sheets ===")
    print(
        f"  selected sheets are tested as {len(th)} Δ x {len(ts)} horizon surfaces; "
        "validation artifacts are selected-bin scoped"
    )
    print(
        "  sheet p-values use the peak over that bin's 2D surface, not the whole node cube"
    )
    print(f"  guard {val.EDGE_GUARD} bars either side, so the p-value floor is 1/(usable+1)")
    print("  nulls the strongest touch-probability deviation per horizon\n")

    done_npz = 0
    done_xlsx = 0
    node_cache: dict[str, dict] = {}
    result_cache: dict[str, dict] = {}
    for row in rows:
        node = by_id[row["node"]]
        try:
            if node["id"] not in node_cache:
                node_cache[node["id"]] = shift.load(ws.shift_cube_path(node["family"], node["id"]))
            cube = node_cache[node["id"]]
            if node["id"] not in result_cache:
                data, feat = artifact_node_feature(cube, ws.min_obs)
                result_cache[node["id"]] = val.validate_node(
                    data,
                    feat,
                    cube["horizons"],
                    cube["Δs"],
                    cube["edges"],
                )
            r = val.sheet_from_node_result(result_cache[node["id"]], row)
            val.save(r, ws.validation_array_path(row), {
                    "node": node["id"],
                    "family": node["family"],
                    "feature": node["feature"],
                    "params": node["params"],
                    "workspace": ws.dir.name,
                    "bin_labels": r["meta"]["bin_labels"],
                    "source_bin": int(row["bin"]),
                    "source_bin_number": int(row["bin_number"]),
                    "selection_rank": int(row["rank"]),
                    "selection_score": float(row["score"]),
                    "generated": datetime.now(timezone.utc).isoformat(),
                })
            done_npz += 1
            writer.write_validation_xlsx(
                r,
                ws.validation_surface_path(row),
                node["id"],
                node["feature"],
                node["params"],
                ws.horizon_unit,
            )
            done_xlsx += 1
        except Exception as e:
            print(f"  {node['id']:<26} [skip] {e}")
            continue

        bins = [int(row["bin"])]
        sheet_p = r["sheet_peak_p"]
        finite = np.isfinite(sheet_p)
        if finite.any():
            rb, k = np.unravel_index(int(np.nanargmin(np.where(finite, sheet_p, np.nan))), sheet_p.shape)
            b = bins[rb]
            floor = 1.0 / (1.0 + r["n_shifts"][k])
            tag = "  [at the floor]" if r["sheet_peak_p"][rb, k] <= floor + 1e-12 else ""
            print(
                f"  {node['id']:<26} bin {b + 1:>2} best p={r['sheet_peak_p'][rb, k]:.5f} "
                f"at +{r['horizons'][k]}{ws.horizon_unit}  peak "
                f"{r['sheet_peak_real'][rb, k]:5.1f} vs p95 "
                f"{r['sheet_peak_p95'][rb, k]:5.1f}{tag}"
            )
        else:
            print(f"  {node['id']:<26} insufficient data in selected bins")

    print(f"\n  wrote {done_npz} .npz artifacts under {ws.dir.relative_to(ROOT)}/04_validation_array/")
    print(f"  wrote {done_xlsx} workbooks under {ws.dir.relative_to(ROOT)}/04_validation_xlsx/")
    print("  verdicts are assigned by --gate across the selected sheet/horizon tests")


def cmd_gate(ws: Workspace, q: float = 0.05) -> None:
    """Stage 5: correct selected sheet tests, then intersect with the economic filter."""
    selected = _selected_rows(ws)
    if not selected:
        print("No 03_selection_array/selection.json - run --selection first.")
        return
    selected = [r for r in selected if ws.has_selection_array(r)]
    if not selected:
        print("No 03_selection_array artifacts - run --selection first.")
        return

    rows = []
    for sel in selected:
        path = ws.validation_array_path(sel)
        if not path.exists():
            continue
        if not _validation_matches_selection(ws, sel):
            continue
        r = val.load(path)
        b = int(sel["bin"])
        for j, t in enumerate(r["horizons"]):
            rows.append({
                "node": sel["node"],
                "family": sel["family"],
                "bin": b,
                "bin_number": b + 1,
                "bin_label": sel.get("bin_label"),
                "selection_rank": sel.get("rank"),
                "selection_score": sel.get("score"),
                "horizon": int(t),
                "peak_p": float(r["sheet_peak_p"][0, j]),
                "peak_real": float(r["sheet_peak_real"][0, j]),
                "peak_p95": float(r["sheet_peak_p95"][0, j]),
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
    n_nom = sum(1 for r in rows if np.isfinite(r["peak_p"]) and r["peak_p"] <= 0.05)
    floors = sorted({r["n_shifts"] for r in rows if r["n_shifts"]})
    floor_lo = 1.0 / (1.0 + max(floors)) if floors else float("nan")

    print(f"\n=== Gate [{ws.dir.name}] ===")
    print(
        f"  {m} tests over {len(selected)} selected sheets x "
        f"{len({r['horizon'] for r in rows})} horizons"
    )
    print(
        f"  Benjamini-Hochberg at q = {q}   |   p-value floor ~ {floor_lo:.2e} "
        "(exact null has only n distinct shifts)"
    )
    print()
    print(f"  BH discoveries    : {n_disc:>4} / {m}")
    print(f"  p<=0.05 before FDR: {n_nom:>4} / {m}")

    econ_passed = {
        (r["node"], int(r["bin"]))
        for r in selected
        if r.get("economic", {}).get("passed")
    }
    by_sheet: dict[tuple[str, int], dict] = {}
    for r in rows:
        if r["verdict"] != "discovery":
            continue
        key = (r["node"], int(r["bin"]))
        cur = by_sheet.get(key)
        if cur is None or r["q_value"] < cur["q_value"]:
            by_sheet[key] = r

    cleared = []
    selected_lookup = {(r["node"], int(r["bin"])): r for r in selected}
    for (nid, b), r in by_sheet.items():
        if (nid, b) not in econ_passed:
            continue
        sel = selected_lookup.get((nid, b), {})
        best = sel.get("economic", {}).get("best") or sel.get("best_cell")
        cleared.append({
            "node": nid,
            "family": r["family"],
            "bin": b,
            "bin_number": b + 1,
            "bin_label": r.get("bin_label"),
            "selection_rank": r.get("selection_rank"),
            "selection_score": r.get("selection_score"),
            "horizon": r["horizon"],
            "q_value": r["q_value"],
            "peak_p": r["peak_p"],
            "peak_real": r["peak_real"],
            "at_floor": r["at_floor"],
            "best_cell": best,
            "selection_cell": sel.get("best_cell"),
            "economic": sel.get("economic"),
        })
    cleared.sort(key=lambda c: (c["q_value"], c.get("selection_rank") or 10**9))

    print(f"\n  selected sheets passing economic filter : {len(econ_passed)}")
    print(f"  selected-sheet discovery and economic pass: {len(cleared)}\n")
    if cleared:
        print(f"  {'node':<26}{'family':<13}{'bin':>5}{'t':>5}  {'q':<10}{'peak':>7}  selected cell")
        print(f"  {'-'*26}{'-'*13}{'-'*5}{'-'*5}  {'-'*10}{'-'*7}  {'-'*36}")
        for c in cleared:
            b = c["best_cell"] or {}
            cell = (
                f"{b.get('dev', 0):+.1f}pp @ Δ={b.get('Δ', 0):+.0%} "
                f"n={b.get('bin_n')}"
            ) if b else ""
            print(
                f"  {c['node']:<26}{c['family']:<13}{c['bin_number']:>5}{c['horizon']:>5}  "
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
            "unit": "selected node/bin sheet x horizon",
            "note": "circular-shift null is exact over all n shifts; the floor 1/(n+1) "
                    "limits how small any reported p-value can be",
        },
        "summary": {"discovery": n_disc, "nominal": n_nom, "cleared": len(cleared)},
        "cleared": cleared,
        "tests": rows,
    })
    print(f"\n  wrote {ws.cleared_path.relative_to(ROOT)}")
