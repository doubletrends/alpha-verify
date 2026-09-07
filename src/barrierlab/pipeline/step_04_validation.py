"""Stage 4: selected-node nulls, global correction, and final verdicts."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from barrierlab.domain import barrier, validation as val
from barrierlab.infrastructure import artifact_io
from barrierlab.infrastructure.artifacts import (
    feature_from_artifact,
    market_data_from_artifact,
)
from barrierlab.infrastructure.workspace import Workspace
from barrierlab.pipeline.context import RunContext
from barrierlab.presentation import workbooks


def validation_artifact_is_current(ws: Workspace, row: dict) -> bool:
    """Return whether a validation array matches its selected-node artifact."""
    if not ws.has_validation_array(row):
        return False
    try:
        selected_node = artifact_io.load_selected_node(ws.selection_array_path(row))
        existing = artifact_io.load_validation(ws.validation_array_path(row))
    except Exception:
        return False
    return (
        np.array_equal(existing["Δs"], selected_node["Δs"])
        and np.array_equal(existing["horizons"], selected_node["horizons"])
        and existing["cell_real"].shape
        == (selected_node["shift"].shape[0], 1, selected_node["shift"].shape[2])
        and int(np.asarray(selected_node.get("source_bin", -1))) == int(row["bin"])
        and int(np.asarray(selected_node.get("selection_rank", -1))) == int(row["rank"])
        and np.isclose(
            float(np.asarray(selected_node.get("selection_score", np.nan))),
            float(row["score"]),
            rtol=1e-6,
            atol=1e-9,
        )
        and int(np.asarray(existing.get("source_bin", -1))) == int(row["bin"])
        and int(np.asarray(existing.get("selection_rank", -1))) == int(row["rank"])
        and np.isclose(
            float(np.asarray(existing.get("selection_score", np.nan))),
            float(row["score"]),
            rtol=1e-6,
            atol=1e-9,
        )
        and "sheet_peak_p" in existing
        and "node_peak_p" in existing
        and "selected_peak_null" in existing
    )


def _selected_rows(ws: Workspace) -> list[dict]:
    selection = ws.read_json(ws.selection_path)
    return selection.get("selected", [])


def selection_fingerprint(selected: list[dict]) -> list[dict]:
    """Stable identity of the selection that a validation summary certifies."""
    return [
        {
            "rank": int(row["rank"]),
            "node": row["node"],
            "bin": int(row["bin"]),
            "score": float(row["score"]),
        }
        for row in selected
    ]


def _economic_criteria(ws: Workspace) -> dict:
    return {
        "min_dev": ws.min_dev,
        "min_bin_n": ws.min_bin_n,
        "min_run": ws.min_run,
        "scope": "selected node's representative bin",
    }


def _evaluate_economics(ws: Workspace, selected: list[dict]) -> list[dict]:
    rows = []
    for row in selected:
        cube = artifact_io.load_selected_node(ws.selection_array_path(row))
        result = val.economic_filter_sheet(
            cube,
            int(row["bin"]),
            ws.min_dev,
            ws.min_bin_n,
            ws.min_run,
        )
        rows.append({
            "node": row["node"],
            "family": row["family"],
            "bin": int(row["bin"]),
            "bin_number": int(row["bin"]) + 1,
            "bin_label": row.get("bin_label"),
            **result,
        })
    return rows


def validation_summary_is_current(ws: Workspace, summary: dict) -> bool:
    """Return whether a complete summary certifies the current selected artifacts."""
    if not summary or not summary.get("complete"):
        return False
    selected = _selected_rows(ws)
    if not selected or summary.get("selection_fingerprint") != selection_fingerprint(selected):
        return False
    if summary.get("method", {}).get("unit") == "one two-sided condition-bin score":
        return True
    if summary.get("method", {}).get("economic_filter") != _economic_criteria(ws):
        return False
    if summary.get("method", {}).get("null_alpha") != ws.null_alpha:
        return False
    if any(
        not ws.has_selection_array(row) or not validation_artifact_is_current(ws, row)
        for row in selected
    ):
        return False
    expected_pairs = {
        (row["node"], int(row["bin"]), int(horizon))
        for row in selected for horizon in ws.horizons
    }
    tests = summary.get("tests", [])
    actual_pairs = {
        (row.get("node"), int(row.get("bin", -1)), int(row.get("horizon", -1)))
        for row in tests
    }
    return len(tests) == len(expected_pairs) and actual_pairs == expected_pairs


def finalize_validation(ws: Workspace, q: float = 0.05) -> bool:
    """Apply global BH and the economic intersection once every null is current."""
    selected = _selected_rows(ws)
    if not selected:
        print("No 03_selection/selection.json - run selection first.")
        return False

    missing = [
        row["node"] for row in selected
        if not ws.has_selection_array(row) or not validation_artifact_is_current(ws, row)
    ]
    if missing:
        ws.write_json(ws.validation_summary_path, {
            "workspace": ws.dir.name,
            "generated": datetime.now(timezone.utc).isoformat(),
            "artifact": "04_validation",
            "complete": False,
            "selection_fingerprint": selection_fingerprint(selected),
            "missing_nodes": missing,
            "method": {
                "correction": "benjamini-hochberg",
                "q": q,
                "null_alpha": ws.null_alpha,
                "unit": "selected node x horizon (peak over all its bins)",
                "economic_filter": _economic_criteria(ws),
            },
            "summary": {
                "tested": 0,
                "null_pass": 0,
                "fdr_pass": 0,
                "discovery": 0,
                "nominal": 0,
                "economic": 0,
                "cleared": 0,
            },
            "tests": [],
            "economics": [],
            "cleared": [],
        })
        print(
            f"\n  validation decision pending: {len(missing)} selected node(s) are missing "
            "current null artifacts"
        )
        print(f"  wrote incomplete {ws.validation_summary_path.relative_to(ws.root_dir)}")
        return False

    rows = []
    for selection in selected:
        result = artifact_io.load_validation(ws.validation_array_path(selection))
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
                "peak_p": float(result["node_peak_p"][index]),
                "peak_real": float(result["node_peak_real"][index]),
                "peak_p95": float(result["node_peak_p95"][index]),
                "n_shifts": int(result["n_shifts"][index]),
            })

    economics = _evaluate_economics(ws, selected)
    economic_by_node = {row["node"]: row for row in economics}
    economic_passes = {row["node"] for row in economics if row["passed"]}

    pvalues = np.array([row["peak_p"] for row in rows])
    rejected, qvalues = val.bh(pvalues, q)
    for row, is_rejected, qvalue in zip(rows, rejected, qvalues):
        floor = 1.0 / (1.0 + row["n_shifts"]) if row["n_shifts"] else np.nan
        row["q_value"] = None if not np.isfinite(qvalue) else round(float(qvalue), 6)
        row["at_floor"] = bool(
            np.isfinite(row["peak_p"]) and np.isfinite(floor)
            and row["peak_p"] <= floor + 1e-12
        )
        row["null_pass"] = bool(
            np.isfinite(row["peak_p"]) and row["peak_p"] <= ws.null_alpha
        )
        row["fdr_pass"] = bool(is_rejected)
        row["economic_pass"] = row["node"] in economic_passes
        row["cleared"] = row["null_pass"] and row["fdr_pass"] and row["economic_pass"]
        row["verdict"] = val.verdict(
            row["fdr_pass"], row["peak_p"], row["at_floor"], row["null_pass"]
        )

    test_count = int(np.isfinite(pvalues).sum())
    null_pass_count = sum(row["null_pass"] for row in rows)
    fdr_pass_count = sum(row["fdr_pass"] for row in rows)
    discovery_count = sum(row["null_pass"] and row["fdr_pass"] for row in rows)
    nominal_count = sum(
        np.isfinite(row["peak_p"]) and row["peak_p"] <= 0.05 for row in rows
    )
    shift_counts = sorted({row["n_shifts"] for row in rows if row["n_shifts"]})
    floor_lo = 1.0 / (1.0 + max(shift_counts)) if shift_counts else float("nan")

    print(f"\n=== 4. Validation decision [{ws.dir.name}] ===")
    print(
        f"  {test_count} tests over {len(selected)} selected nodes x "
        f"{len({row['horizon'] for row in rows})} horizons"
    )
    print(
        f"  raw node-null alpha = {ws.null_alpha}   |   Benjamini-Hochberg q = {q}"
    )
    print(
        f"  p-value floor ~ {floor_lo:.2e} "
        "(exact null has only n distinct shifts)"
    )
    print()
    print(f"  raw null passes   : {null_pass_count:>4} / {test_count}")
    print(f"  BH/FDR passes     : {fdr_pass_count:>4} / {test_count}")
    print(f"  raw + FDR passes  : {discovery_count:>4} / {test_count}")
    print(f"  p<=0.05 before FDR: {nominal_count:>4} / {test_count}")

    by_node: dict[str, dict] = {}
    for row in rows:
        if not row["cleared"]:
            continue
        current = by_node.get(row["node"])
        if current is None or row["q_value"] < current["q_value"]:
            by_node[row["node"]] = row

    selected_lookup = {row["node"]: row for row in selected}
    cleared = []
    for node_id, row in by_node.items():
        selection = selected_lookup[node_id]
        bin_index = int(selection["bin"])
        economic = economic_by_node[node_id]
        if node_id not in economic_passes:
            continue
        best_cell = economic.get("best")
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
            "null_pass": row["null_pass"],
            "fdr_pass": row["fdr_pass"],
            "economic_pass": economic["passed"],
            "cleared": True,
            "best_cell": best_cell,
            "selection_cell": selection.get("best_cell"),
            "economic": economic,
        })
    cleared.sort(key=lambda item: (item["q_value"], item.get("selection_rank") or 10**9))

    print(f"\n  selected representative bins passing economic filter : {len(economic_passes)}")
    print(f"  all three filters passed: {len(cleared)} selected nodes\n")
    if cleared:
        print(
            f"  {'node':<26}{'family':<13}{'bin':>5}{'t':>5}  "
            f"{'p':<10}{'q':<10}{'peak':>7}  selected cell"
        )
        print(
            f"  {'-' * 26}{'-' * 13}{'-' * 5}{'-' * 5}  "
            f"{'-' * 10}{'-' * 10}{'-' * 7}  {'-' * 36}"
        )
        for item in cleared:
            best_cell = item["best_cell"] or {}
            cell = (
                f"{best_cell.get('dev', 0):+.1f}pp @ Delta={best_cell.get('Δ', 0):+.0%} "
                f"n={best_cell.get('bin_n')}"
            ) if best_cell else ""
            print(
                f"  {item['node']:<26}{item['family']:<13}{item['bin_number']:>5}{item['horizon']:>5}  "
                f"{item['peak_p']:<10.5f}{item['q_value']:<10.5f}"
                f"{item['peak_real']:>7.1f}  {cell}"
            )
    else:
        print("  Nothing clears all three filters.")

    summary = {
        "workspace": ws.dir.name,
        "generated": datetime.now(timezone.utc).isoformat(),
        "artifact": "04_validation",
        "complete": True,
        "selection_fingerprint": selection_fingerprint(selected),
        "missing_nodes": [],
        "method": {
            "correction": "benjamini-hochberg",
            "q": q,
            "null_alpha": ws.null_alpha,
            "n_tests": test_count,
            "p_floor": floor_lo,
            "unit": "selected node x horizon (peak over all its bins)",
            "economic_filter": _economic_criteria(ws),
            "note": "circular-shift null is exact over all n shifts; the floor 1/(n+1) "
                    "limits how small any reported p-value can be",
        },
        "summary": {
            "tested": test_count,
            "null_pass": null_pass_count,
            "fdr_pass": fdr_pass_count,
            "discovery": discovery_count,
            "nominal": nominal_count,
            "economic": len(economic_passes),
            "cleared": len(cleared),
        },
        "economics": economics,
        "cleared": cleared,
        "tests": rows,
    }
    ws.write_json(ws.validation_summary_path, summary)
    print(f"\n  wrote {ws.validation_summary_path.relative_to(ws.root_dir)}")
    return True


def cmd_validation(ws: Workspace) -> None:
    """Run selected-node nulls, then finalize correction and economic verdicts."""
    rows = _selected_rows(ws)
    if not rows:
        print("No selected nodes - run selection first.")
        return
    rows = [row for row in rows if ws.has_selection_array(row)]
    if not rows:
        print("No 03_selection artifacts - run selection first.")
        return
    nodes = {node["id"]: node for node in ws.catalog.all_nodes()}
    rows = [row for row in rows if row["node"] in nodes]
    print(f"\n=== 4. Validation [{ws.dir.name}] — selected-bin skew ===")
    context = RunContext(ws)
    grouped, simulated_paths = {}, None
    for row in rows:
        selected_node = artifact_io.load_selected_node(ws.selection_array_path(row))
        data = market_data_from_artifact(selected_node)
        feature = feature_from_artifact(selected_node, data.index)
        delta, horizon = ws.target(selected_node["base"])
        if simulated_paths is None:
            simulated_paths = val.simulated_ohlc_tensor(data)
        node = nodes[row["node"]]
        grouped.setdefault(row["node"], {"node": node, "data": data, "feature": feature,
                                           "delta": delta, "horizon": horizon, "rows": []})["rows"].append(row)

    records = []
    for group in grouped.values():
        node, data, feature = group["node"], group["data"], group["feature"]
        fixed_edges = None if node["data"] == ["ohlcv"] else barrier.bin_edges(feature, ws.n_bins)
        observed_scores = val.bin_scores(
            data, feature, delta, horizon, ws.n_bins,
            baseline=val.baseline_prob(data, delta, horizon), edges=fixed_edges,
        )
        synthetic_features = None
        if node["data"] != ["ohlcv"]:
            synthetic_features = np.broadcast_to(feature.to_numpy(float), (simulated_paths.shape[0], len(feature)))
        null_matrix = val.batched_bin_scores(
            simulated_paths, synthetic_features, delta, horizon, ws.n_bins,
            node["feature"], node["params"],
        )
        for row in group["rows"]:
            observed = float(observed_scores[int(row["bin"])])
            null_scores = null_matrix[:, int(row["bin"])]
            records.append({**row, "horizon": horizon, "bin_score": observed,
                            "peak_p": float((1 + (null_scores >= observed).sum()) / (1 + len(null_scores))),
                            "peak_p95": float(np.percentile(null_scores, 95)),
                            "n_shifts": len(null_scores),
                            "null_scores": [float(value) for value in null_scores]})
    rejected, qvalues = val.bh(np.asarray([r["peak_p"] for r in records]))
    cleared = []
    for row, rejected_here, qvalue in zip(records, rejected, qvalues):
        row["q_value"] = float(qvalue)
        row["cleared"] = bool(rejected_here and row["peak_p"] <= ws.null_alpha)
        if row["cleared"]:
            cleared.append({key: row[key] for key in (
                "node", "family", "rank", "score", "horizon", "bin_score",
                "peak_p", "peak_p95", "n_shifts", "q_value", "cleared",
            )})
    summary = {
        "workspace": ws.dir.name, "generated": datetime.now(timezone.utc).isoformat(),
        "artifact": "04_validation", "complete": True,
        "selection_fingerprint": selection_fingerprint(rows),
        "method": {"correction": "benjamini-hochberg", "q": 0.05,
                   "null_alpha": ws.null_alpha,
                   "unit": "one two-sided condition-bin score",
                   "null": "1000 shared synthetic OHLC histories; external condition histories fixed"},
        "summary": {"tested": len(records), "cleared": len(cleared)},
        "tests": records, "cleared": cleared, "economics": [],
    }
    ws.write_json(ws.validation_summary_path, summary)
    from barrierlab.presentation.validation_plots import write_bin_score_null_histograms
    write_bin_score_null_histograms(ws, summary)
    print(f"  tested {len(records)} selected-bin scores; {len(cleared)} cleared after BH")
