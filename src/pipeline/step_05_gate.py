"""Stage 5: false-discovery correction and economic intersection."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from domain import validation as val
from infrastructure.workspaces.workspace import Workspace
from pipeline.step_04_validation import (
    _selected_rows,
    _validation_matches_selection,
)


def cmd_gate(ws: Workspace, q: float = 0.05) -> None:
    """Correct selected-sheet tests, then intersect with the economic filter."""
    selected = _selected_rows(ws)
    if not selected:
        print("No 03_selection_array/selection.json - run --selection first.")
        return
    selected = [row for row in selected if ws.has_selection_array(row)]
    if not selected:
        print("No 03_selection_array artifacts - run --selection first.")
        return

    rows = []
    for selection in selected:
        path = ws.validation_array_path(selection)
        if not path.exists() or not _validation_matches_selection(ws, selection):
            continue
        result = val.load(path)
        source_bin = int(selection["bin"])
        for index, horizon in enumerate(result["horizons"]):
            rows.append({
                "node": selection["node"],
                "family": selection["family"],
                "bin": source_bin,
                "bin_number": source_bin + 1,
                "bin_label": selection.get("bin_label"),
                "selection_rank": selection.get("rank"),
                "selection_score": selection.get("score"),
                "horizon": int(horizon),
                "peak_p": float(result["sheet_peak_p"][0, index]),
                "peak_real": float(result["sheet_peak_real"][0, index]),
                "peak_p95": float(result["sheet_peak_p95"][0, index]),
                "n_shifts": int(result["n_shifts"][index]),
            })
    if not rows:
        print("No validation artifacts - run --validation first.")
        return

    pvalues = np.array([row["peak_p"] for row in rows])
    rejected, qvalues = val.bh(pvalues, q)
    for row, is_rejected, qvalue in zip(rows, rejected, qvalues):
        floor = 1.0 / (1.0 + row["n_shifts"]) if row["n_shifts"] else np.nan
        row["q_value"] = None if not np.isfinite(qvalue) else round(float(qvalue), 6)
        row["at_floor"] = bool(
            np.isfinite(row["peak_p"]) and np.isfinite(floor)
            and row["peak_p"] <= floor + 1e-12
        )
        row["verdict"] = val.verdict(bool(is_rejected), row["peak_p"], row["at_floor"])

    test_count = int(np.isfinite(pvalues).sum())
    discovery_count = sum(row["verdict"] == "discovery" for row in rows)
    nominal_count = sum(
        np.isfinite(row["peak_p"]) and row["peak_p"] <= 0.05 for row in rows
    )
    shift_counts = sorted({row["n_shifts"] for row in rows if row["n_shifts"]})
    floor_lo = 1.0 / (1.0 + max(shift_counts)) if shift_counts else float("nan")

    print(f"\n=== Gate [{ws.dir.name}] ===")
    print(
        f"  {test_count} tests over {len(selected)} selected sheets x "
        f"{len({row['horizon'] for row in rows})} horizons"
    )
    print(
        f"  Benjamini-Hochberg at q = {q}   |   p-value floor ~ {floor_lo:.2e} "
        "(exact null has only n distinct shifts)"
    )
    print()
    print(f"  BH discoveries    : {discovery_count:>4} / {test_count}")
    print(f"  p<=0.05 before FDR: {nominal_count:>4} / {test_count}")

    economic_passes = {
        (row["node"], int(row["bin"]))
        for row in selected if row.get("economic", {}).get("passed")
    }
    by_sheet: dict[tuple[str, int], dict] = {}
    for row in rows:
        if row["verdict"] != "discovery":
            continue
        key = (row["node"], int(row["bin"]))
        current = by_sheet.get(key)
        if current is None or row["q_value"] < current["q_value"]:
            by_sheet[key] = row

    cleared = []
    selected_lookup = {(row["node"], int(row["bin"])): row for row in selected}
    for (node_id, bin_index), row in by_sheet.items():
        if (node_id, bin_index) not in economic_passes:
            continue
        selection = selected_lookup.get((node_id, bin_index), {})
        best_cell = selection.get("economic", {}).get("best") or selection.get("best_cell")
        cleared.append({
            "node": node_id,
            "family": row["family"],
            "bin": bin_index,
            "bin_number": bin_index + 1,
            "bin_label": row.get("bin_label"),
            "selection_rank": row.get("selection_rank"),
            "selection_score": row.get("selection_score"),
            "horizon": row["horizon"],
            "q_value": row["q_value"],
            "peak_p": row["peak_p"],
            "peak_real": row["peak_real"],
            "at_floor": row["at_floor"],
            "best_cell": best_cell,
            "selection_cell": selection.get("best_cell"),
            "economic": selection.get("economic"),
        })
    cleared.sort(key=lambda item: (item["q_value"], item.get("selection_rank") or 10**9))

    print(f"\n  selected sheets passing economic filter : {len(economic_passes)}")
    print(f"  selected-sheet discovery and economic pass: {len(cleared)}\n")
    if cleared:
        print(f"  {'node':<26}{'family':<13}{'bin':>5}{'t':>5}  {'q':<10}{'peak':>7}  selected cell")
        print(f"  {'-' * 26}{'-' * 13}{'-' * 5}{'-' * 5}  {'-' * 10}{'-' * 7}  {'-' * 36}")
        for item in cleared:
            best_cell = item["best_cell"] or {}
            cell = (
                f"{best_cell.get('dev', 0):+.1f}pp @ Delta={best_cell.get('Δ', 0):+.0%} "
                f"n={best_cell.get('bin_n')}"
            ) if best_cell else ""
            print(
                f"  {item['node']:<26}{item['family']:<13}{item['bin_number']:>5}{item['horizon']:>5}  "
                f"{item['q_value']:<10.5f}{item['peak_real']:>7.1f}  {cell}"
            )
    else:
        print("  Nothing clears both filters.")

    ws.write_json(ws.cleared_path, {
        "workspace": ws.dir.name,
        "generated": datetime.now(timezone.utc).isoformat(),
        "method": {
            "correction": "benjamini-hochberg",
            "q": q,
            "n_tests": test_count,
            "p_floor": floor_lo,
            "unit": "selected node/bin sheet x horizon",
            "note": "circular-shift null is exact over all n shifts; the floor 1/(n+1) "
                    "limits how small any reported p-value can be",
        },
        "summary": {
            "discovery": discovery_count,
            "nominal": nominal_count,
            "cleared": len(cleared),
        },
        "cleared": cleared,
        "tests": rows,
    })
    print(f"\n  wrote {ws.cleared_path.relative_to(ws.root_dir)}")
