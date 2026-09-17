"""Stage 5: which nodes cleared, and with which condition bins."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone

from alphaverify.infrastructure import artifact_io
from alphaverify.infrastructure.workspace import Workspace
from alphaverify.pipeline.reporting import StageReport
from alphaverify.pipeline.step_04_selection import selection_summary_is_current
from alphaverify.presentation import workbooks

_BIN_FIELDS = ("bin", "bin_number", "bin_label", "rank", "monte_carlo_p_value", "q_value")


def node_summary_is_current(ws: Workspace, summary: dict) -> bool:
    """Require a complete summary sourced from the current selection bytes."""
    selection = artifact_io.read_json(ws.selection_summary_path)
    if (not summary or not summary.get("complete")
            or summary.get("artifact") != "05_summary"
            or not selection_summary_is_current(ws, selection)):
        return False
    return summary.get("selection_sha256") == artifact_io.file_sha256(ws.selection_summary_path)


def cleared_nodes(selected: list[dict], tested_bins: Counter) -> list[dict]:
    """Group ranked selected bins by node; nodes follow their strongest bin's rank."""
    nodes: dict[str, dict] = {}
    for row in selected:
        node = nodes.setdefault(row["node"], {
            "node": row["node"], "family": row["family"],
            "feature": row["feature"], "params": row["params"],
            "bins_tested": tested_bins[row["node"]], "bins_cleared": 0, "bins": [],
        })
        node["bins_cleared"] += 1
        node["bins"].append({field: row[field] for field in _BIN_FIELDS})
    for node in nodes.values():
        node["bins"].sort(key=lambda item: item["bin"])
        node["best_rank"] = min(item["rank"] for item in node["bins"])
        node["best_p_value"] = min(item["monte_carlo_p_value"] for item in node["bins"])
        node["best_q_value"] = min(item["q_value"] for item in node["bins"])
    return sorted(nodes.values(), key=lambda node: (node["best_rank"], node["node"]))


def cmd_summary(ws: Workspace) -> None:
    """Write the cleared-node manifest and its workbook view."""
    report = StageReport(5)
    selection = artifact_io.read_json(ws.selection_summary_path)
    if not selection_summary_is_current(ws, selection):
        report.line("selection is missing or stale; run select first")
        report.completed()
        return

    selection_sha256 = artifact_io.file_sha256(ws.selection_summary_path)
    validation = artifact_io.read_json(ws.validation_summary_path)
    tested_bins = Counter(row["node"] for row in validation.get("tests", []))
    report.line(f"grouping {len(selection.get('selected', []))} selected bins by node")
    nodes = cleared_nodes(selection.get("selected", []), tested_bins)
    bins_cleared = sum(node["bins_cleared"] for node in nodes)
    summary = {
        "workspace": ws.dir.name,
        "generated": datetime.now(timezone.utc).isoformat(),
        "artifact": "05_summary",
        "complete": True,
        "source": ws.selection_summary_path.relative_to(ws.dir).as_posix(),
        "selection_sha256": selection_sha256,
        "summary": {
            "nodes_tested": len(tested_bins),
            "nodes_cleared": len(nodes),
            "bins_tested": sum(tested_bins.values()),
            "bins_cleared": bins_cleared,
            "expected_by_chance": selection["summary"]["expected_by_chance"],
        },
        "nodes": nodes,
    }
    if artifact_io.file_sha256(ws.selection_summary_path) != selection_sha256:
        raise RuntimeError("Stage 4 selection changed during summary; rerun summarize")
    artifact_io.write_json(ws.node_summary_path, summary)

    warnings = []
    try:
        workbooks.write_node_summary_xlsx(nodes, ws.node_summary_workbook_path)
    except PermissionError:
        warnings.append("summary workbook locked; close it in Excel and re-run")
    report.summary(
        f"{len(nodes)} of {len(tested_bins)} tested nodes cleared at least one bin; "
        f"{bins_cleared} bins cleared, about "
        f"{summary['summary']['expected_by_chance']:.1f} expected by chance"
    )
    workbook = 0 if warnings else 1
    report.line(
        f"wrote 1 summary + {workbook} workbook → {ws.stage_dir('summary').relative_to(ws.root_dir)}"
    )
    report.completed()
    StageReport.warnings(warnings)
