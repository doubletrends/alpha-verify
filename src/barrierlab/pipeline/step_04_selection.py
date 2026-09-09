"""Stage 4: retain exactly the condition bins that cleared Stage 3."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib

from barrierlab.infrastructure.workspace import Workspace
from barrierlab.pipeline.context import materialized_shift
from barrierlab.pipeline.reporting import MilestoneProgress, StageReport
from barrierlab.pipeline.step_03_validation import validation_summary_is_current


def _file_sha256(path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def selection_summary_is_current(ws: Workspace, selection: dict) -> bool:
    """Require a complete selection sourced from the current validation bytes."""
    validation = ws.read_json(ws.validation_summary_path)
    if (not selection or not selection.get("complete")
            or selection.get("artifact") != "04_selection"
            or not validation_summary_is_current(ws, validation)
            or not ws.validation_summary_path.exists()):
        return False
    return selection.get("validation_sha256") == _file_sha256(ws.validation_summary_path)


def cmd_selection(ws: Workspace) -> None:
    """Write the cleared-bin manifest and one full shift heatmap per bin."""
    report = StageReport(4, "select", ws.dir.name)
    validation = ws.read_json(ws.validation_summary_path)
    if not validation_summary_is_current(ws, validation):
        report.line("validation is missing, incomplete, or stale; run validate first")
        report.completed()
        return

    validation_sha256 = _file_sha256(ws.validation_summary_path)
    selected = [dict(row) for row in validation.get("cleared", []) if row.get("cleared")]
    selected.sort(key=lambda row: (
        float(row["monte_carlo_p_value"]), -float(row["bin_score"]),
        row["node"], int(row["bin"]),
    ))
    for number, row in enumerate(selected, 1):
        row["selection_number"] = number

    summary = {
        "workspace": ws.dir.name,
        "generated": datetime.now(timezone.utc).isoformat(),
        "artifact": "04_selection",
        "complete": True,
        "source": "03_validation/validation.json",
        "validation_sha256": validation_sha256,
        "method": {
            "rule": "retain every Stage 3 condition bin with cleared == true",
            "threshold": validation["method"]["threshold"],
            "ordering": "ascending raw p, descending observed bin score, stable identity",
        },
        "summary": {
            "bins": len(selected),
            "nodes": len({row["node"] for row in selected}),
        },
        "selected": selected,
    }
    if _file_sha256(ws.validation_summary_path) != validation_sha256:
        raise RuntimeError("Stage 3 validation changed during selection; rerun select")
    ws.write_json(ws.selection_summary_path, summary)

    from barrierlab.presentation.selection_plots import (
        copy_selected_null_histograms, write_selected_shift_heatmaps,
    )
    cubes = {node: materialized_shift(ws, node) for node in {row["node"] for row in selected}}
    plots = write_selected_shift_heatmaps(
        ws, selected, cubes,
        MilestoneProgress(report, "writing selected-bin heatmaps", len(selected)),
    )
    distributions, missing_distributions = copy_selected_null_histograms(ws, selected)
    report.summary(
        f"selected {len(selected)} cleared bins across {summary['summary']['nodes']} nodes"
    )
    report.line(
        f"wrote 1 manifest + {len(plots)} heatmaps + {len(distributions)} null distributions "
        f"→ {ws.selection_summary_path.parent}"
    )
    if missing_distributions:
        report.line(
            f"{len(missing_distributions)} selected null distributions unavailable; rerun validate"
        )
    report.completed()
