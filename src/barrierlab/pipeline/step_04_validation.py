"""Stage 4: validate selected condition-bin skew with simulated OHLC histories."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from barrierlab.domain import validation as val
from barrierlab.domain.features import is_ohlcv_feature
from barrierlab.infrastructure import artifact_io
from barrierlab.infrastructure.artifacts import feature_from_artifact, market_data_from_artifact
from barrierlab.infrastructure.workspace import Workspace
from barrierlab.pipeline.reporting import MilestoneProgress, StageReport

RAW_P_THRESHOLD = 0.05


def _selected_rows(ws: Workspace) -> list[dict]:
    return ws.read_json(ws.selection_path).get("selected", [])


def selection_fingerprint(selected: list[dict]) -> list[dict]:
    """Stable identity of the selection that a validation summary certifies."""
    return [
        {"rank": int(row["rank"]), "node": row["node"], "bin": int(row["bin"]),
         "delta": float(row["delta"]), "horizon": int(row["horizon"]),
         "score": float(row["score"])}
        for row in selected
    ]


def validation_summary_is_current(ws: Workspace, summary: dict) -> bool:
    """Return whether a complete simulated-OHLC summary matches current selection."""
    selected = _selected_rows(ws)
    try:
        return bool(
            summary
            and summary.get("complete")
            and summary.get("selection_fingerprint") == selection_fingerprint(selected)
            and summary.get("method", {}).get("unit") == "one full-grid linearly Δ-weighted condition-bin score; raw p < 0.05"
        )
    except KeyError:
        # A pre-grid selection manifest cannot certify a grid-based validation.
        return False


def cmd_validation(ws: Workspace) -> None:
    """Validate selected full-grid bin scores against 10,000 null histories."""
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
    report.line(f"testing {len(rows)} selected full-grid bin scores against a simulated-OHLC null")
    grouped, simulated_paths = {}, None
    for row in rows:
        selected_node = artifact_io.load_selected_node(ws.selection_array_path(row))
        data = market_data_from_artifact(selected_node)
        feature = feature_from_artifact(selected_node, data.index)
        if simulated_paths is None:
            simulated_paths = val.simulated_ohlc_tensor(data)
        node = nodes[row["node"]]
        grouped.setdefault(row["node"], {
            "node": node, "data": data, "feature": feature, "cube": selected_node, "rows": [],
        })["rows"].append(row)

    records = []
    null_progress = MilestoneProgress(
        report,
        "scoring synthetic nulls",
        sum(len(group["cube"]["horizons"]) for group in grouped.values()),
    )
    for group in grouped.values():
        node, data, feature, cube = group["node"], group["data"], group["feature"], group["cube"]
        n_bins = int(cube["shift"].shape[1])
        synthetic_features = None
        # Only core OHLCV transforms can be evaluated from the unlabeled
        # synthetic paths.  Calendar and workspace-plugin features have no
        # synthetic equivalent, so retain their observed condition history for
        # every null path (as we do for external data features).
        if not is_ohlcv_feature(node["feature"]):
            synthetic_features = np.broadcast_to(
                feature.to_numpy(float), (simulated_paths.shape[0], len(feature))
            )
        observed_scores = val.bin_score_from_shift_cube(cube)
        null_scores_by_bin = val.batched_bin_score_sums(
            simulated_paths, synthetic_features, cube["Δs"], cube["horizons"], n_bins,
            node["feature"], node["params"], null_progress,
        )
        for row in group["rows"]:
            observed = float(observed_scores[int(row["bin"])])
            null_scores = null_scores_by_bin[:, int(row["bin"])]
            records.append({
                **row, "bin_score": observed,
                "peak_p": float((1 + (null_scores >= observed).sum()) / (1 + len(null_scores))),
                "null_p95": float(np.percentile(null_scores, 95)), "n_shifts": len(null_scores),
                "null_scores": [float(value) for value in null_scores],
            })

    cleared = []
    for row in records:
        row["cleared"] = bool(row["peak_p"] < RAW_P_THRESHOLD)
        if row["cleared"]:
            cleared.append({key: row[key] for key in (
                "node", "family", "rank", "score", "delta", "horizon", "bin_score", "peak_p",
                "null_p95", "n_shifts", "cleared",
            )})
    summary = {
        "workspace": ws.dir.name, "generated": datetime.now(timezone.utc).isoformat(),
        "artifact": "04_validation", "complete": True,
        "selection_fingerprint": selection_fingerprint(rows),
        "method": {
            "threshold": {"raw_p": "< 0.05"},
            "unit": "one full-grid linearly Δ-weighted condition-bin score; raw p < 0.05",
            "null": "10,000 shared synthetic OHLC histories; external condition histories fixed",
        },
        "summary": {"tested": len(records), "cleared": len(cleared)},
        "tests": records, "cleared": cleared,
    }
    ws.write_json(ws.validation_summary_path, summary)
    from barrierlab.presentation.validation_plots import write_bin_score_null_histograms
    plots = write_bin_score_null_histograms(
        ws, summary, MilestoneProgress(report, "writing plots", len(records))
    )
    n_shifts = records[0]["n_shifts"] if records else 0
    report.summary(
        f"cleared {len(cleared)} of {len(records)} with raw p < 0.05 "
        f"({n_shifts:,} null histories)"
    )
    report.line(
        f"wrote 1 summary + {len(plots)} plots → "
        f"{ws.dir.relative_to(ws.root_dir)}/04_validation"
    )
    report.completed()
