"""Stage 4: validate selected condition-bin skew with simulated OHLC histories."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from barrierlab.domain import barrier, validation as val
from barrierlab.domain.features import is_ohlcv_feature
from barrierlab.infrastructure import artifact_io
from barrierlab.infrastructure.artifacts import feature_from_artifact, market_data_from_artifact
from barrierlab.infrastructure.workspace import Workspace
from barrierlab.pipeline.reporting import StageReport


def _selected_rows(ws: Workspace) -> list[dict]:
    return ws.read_json(ws.selection_path).get("selected", [])


def selection_fingerprint(selected: list[dict]) -> list[dict]:
    """Stable identity of the selection that a validation summary certifies."""
    return [
        {"rank": int(row["rank"]), "node": row["node"], "bin": int(row["bin"]),
         "score": float(row["score"])}
        for row in selected
    ]


def validation_summary_is_current(ws: Workspace, summary: dict) -> bool:
    """Return whether a complete simulated-OHLC summary matches current selection."""
    selected = _selected_rows(ws)
    return bool(
        summary
        and summary.get("complete")
        and summary.get("selection_fingerprint") == selection_fingerprint(selected)
        and summary.get("method", {}).get("unit") == "one two-sided condition-bin score"
    )


def cmd_validation(ws: Workspace) -> None:
    """Validate selected-bin skew against 1,000 shared synthetic OHLC histories."""
    rows = [row for row in _selected_rows(ws) if ws.has_selection_array(row)]
    report = StageReport(4, "validate", ws.dir.name)
    if not rows:
        report.line("no 03_selection artifacts available; run select first")
        report.completed()
        return

    nodes = {node["id"]: node for node in ws.catalog.all_nodes()}
    rows = [row for row in rows if row["node"] in nodes]
    if not rows:
        report.line("no selected nodes are declared in this workspace")
        report.completed()
        return
    report.line(f"testing {len(rows)} selected-bin scores against a simulated-OHLC null")
    grouped, simulated_paths = {}, None
    for row in rows:
        selected_node = artifact_io.load_selected_node(ws.selection_array_path(row))
        data = market_data_from_artifact(selected_node)
        feature = feature_from_artifact(selected_node, data.index)
        delta, horizon = ws.target(selected_node["base"])
        if simulated_paths is None:
            simulated_paths = val.simulated_ohlc_tensor(data)
        node = nodes[row["node"]]
        grouped.setdefault(row["node"], {
            "node": node, "data": data, "feature": feature, "delta": delta,
            "horizon": horizon, "rows": [],
        })["rows"].append(row)

    records = []
    for group in grouped.values():
        node, data, feature = group["node"], group["data"], group["feature"]
        delta, horizon = group["delta"], group["horizon"]
        fixed_edges = None if node["data"] == ["ohlcv"] else barrier.bin_edges(feature, ws.n_bins)
        observed_scores = val.bin_scores(
            data, feature, delta, horizon, ws.n_bins,
            baseline=val.baseline_prob(data, delta, horizon), edges=fixed_edges,
        )
        synthetic_features = None
        # Only core OHLCV transforms can be evaluated from the unlabeled
        # synthetic paths.  Calendar and workspace-plugin features have no
        # synthetic equivalent, so retain their observed condition history for
        # every null path (as we do for external data features).
        if not is_ohlcv_feature(node["feature"]):
            synthetic_features = np.broadcast_to(
                feature.to_numpy(float), (simulated_paths.shape[0], len(feature))
            )
        null_matrix = val.batched_bin_scores(
            simulated_paths, synthetic_features, delta, horizon, ws.n_bins,
            node["feature"], node["params"],
        )
        for row in group["rows"]:
            observed = float(observed_scores[int(row["bin"])])
            null_scores = null_matrix[:, int(row["bin"])]
            records.append({
                **row, "horizon": horizon, "bin_score": observed,
                "peak_p": float((1 + (null_scores >= observed).sum()) / (1 + len(null_scores))),
                "peak_p95": float(np.percentile(null_scores, 95)), "n_shifts": len(null_scores),
                "null_scores": [float(value) for value in null_scores],
            })

    rejected, qvalues = val.bh(np.asarray([row["peak_p"] for row in records]))
    cleared = []
    for row, rejected_here, qvalue in zip(records, rejected, qvalues):
        row["q_value"] = float(qvalue)
        row["cleared"] = bool(rejected_here and row["peak_p"] <= ws.null_alpha)
        if row["cleared"]:
            cleared.append({key: row[key] for key in (
                "node", "family", "rank", "score", "horizon", "bin_score", "peak_p",
                "peak_p95", "n_shifts", "q_value", "cleared",
            )})
    summary = {
        "workspace": ws.dir.name, "generated": datetime.now(timezone.utc).isoformat(),
        "artifact": "04_validation", "complete": True,
        "selection_fingerprint": selection_fingerprint(rows),
        "method": {
            "correction": "benjamini-hochberg", "q": 0.05, "null_alpha": ws.null_alpha,
            "unit": "one two-sided condition-bin score",
            "null": "1000 shared synthetic OHLC histories; external condition histories fixed",
        },
        "summary": {"tested": len(records), "cleared": len(cleared)},
        "tests": records, "cleared": cleared,
    }
    ws.write_json(ws.validation_summary_path, summary)
    from barrierlab.presentation.validation_plots import write_bin_score_null_histograms
    plots = write_bin_score_null_histograms(ws, summary)
    n_shifts = records[0]["n_shifts"] if records else 0
    report.summary(
        f"cleared {len(cleared)} of {len(records)} after Benjamini–Hochberg "
        f"(q={summary['method']['q']}, {n_shifts:,} null histories)"
    )
    report.line(
        f"wrote 1 summary + {len(plots)} plots → "
        f"{ws.dir.relative_to(ws.root_dir)}/04_validation"
    )
    report.completed()
