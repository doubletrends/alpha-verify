"""Stage 4: retain exactly the condition bins that cleared Stage 3."""

from __future__ import annotations

from datetime import datetime, timezone

from alphaverify.infrastructure import artifact_io
from alphaverify.infrastructure.workspace import Workspace
from alphaverify.pipeline.context import materialized_shift
from alphaverify.pipeline.reporting import MilestoneProgress, StageReport
from alphaverify.pipeline.step_03_validation import validation_summary_is_current


def selection_summary_is_current(ws: Workspace, selection: dict) -> bool:
    """Require a complete selection sourced from the current validation bytes."""
    validation = artifact_io.read_json(ws.validation_summary_path)
    if (not selection or not selection.get("complete")
            or selection.get("artifact") != "04_selection"
            or not validation_summary_is_current(ws, validation)):
        return False
    return selection.get("validation_sha256") == artifact_io.file_sha256(ws.validation_summary_path)


def cmd_selection(ws: Workspace) -> None:
    """Write the cleared-bin manifest and one full shift heatmap per bin."""
    report = StageReport(4)
    validation = artifact_io.read_json(ws.validation_summary_path)
    if not validation_summary_is_current(ws, validation):
        report.line("validation is missing, incomplete, or stale; run validate first")
        report.completed()
        return

    validation_sha256 = artifact_io.file_sha256(ws.validation_summary_path)
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
        "source": ws.validation_summary_path.relative_to(ws.dir).as_posix(),
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
    if artifact_io.file_sha256(ws.validation_summary_path) != validation_sha256:
        raise RuntimeError("Stage 3 validation changed during selection; rerun select")
    artifact_io.write_json(ws.selection_summary_path, summary)

    # Deferred so commands that write no figures do not pay matplotlib's import.
    from alphaverify.presentation import selection_plots
    cubes = {node: materialized_shift(ws, node) for node in {row["node"] for row in selected}}
    plots = selection_plots.write_selected_shift_heatmaps(
        ws, selected, cubes,
        MilestoneProgress(report, "writing selected-bin heatmaps", len(selected)),
    )
    distributions, missing_distributions = selection_plots.copy_selected_null_histograms(ws, selected)
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
