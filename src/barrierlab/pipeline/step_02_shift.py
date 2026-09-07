"""Stage 2: baseline-subtracted shift artifacts and workbooks."""

from __future__ import annotations

from barrierlab.domain import shift
from barrierlab.infrastructure import artifact_io
from barrierlab.infrastructure.workspace import BASELINE_NODE, Workspace
from barrierlab.pipeline.context import baseline_surface
from barrierlab.pipeline.reporting import MilestoneProgress, StageReport
from barrierlab.presentation import workbooks


def _write_shift_array(ws: Workspace, progress: MilestoneProgress) -> list[str]:
    nodes = [
        node for node in ws.catalog.all_nodes()
        if ws.has_cube(node["id"])
    ]
    if not nodes:
        return ["no full surface arrays available; run measure first"]

    baseline = baseline_surface(ws)
    if baseline is None:
        return ["no baseline surface array available; run measure first"]

    nodes = [node for node in nodes if node["id"] == BASELINE_NODE] + [
        node for node in nodes if node["id"] != BASELINE_NODE
    ]
    deltas, horizons = ws.deltas, ws.horizons
    skipped = {}
    for node in nodes:
        try:
            full = artifact_io.load_surface(ws.cube_path(node["id"]))
            shifted = shift.from_cube(full, baseline)
            artifact_io.save_shift(
                shifted,
                ws.shift_cube_path(node["id"]),
                {**full["meta"], "grid": "shift", "value": "conditional_minus_baseline_pp"},
            )
        except Exception as error:
            skipped[node["id"]] = str(error)
        finally:
            progress.advance()
    return [f"shift skipped {node}: {error}" for node, error in skipped.items()]


def _render_shift(ws: Workspace, progress: MilestoneProgress) -> tuple[int, list[str]]:
    nodes = [
        node for node in ws.catalog.all_nodes()
        if ws.has_shift_cube(node["id"])
    ]
    if not nodes:
        return 0, ["no shift arrays available; run compare first"]
    written = 0
    warnings = []
    for node in nodes:
        cube = artifact_io.load_shift(ws.shift_cube_path(node["id"]))
        try:
            workbooks.write_shift_xlsx(
                cube,
                ws.shift_surface_path(node["id"]),
                node["id"],
                node["feature"],
                node["params"],
                ws.horizon_unit,
            )
        except PermissionError:
            warnings.append(f"workbook locked for {node['id']}; close it in Excel and re-run")
        else:
            written += 1
        finally:
            progress.advance()
    return written, warnings


def cmd_shift(ws: Workspace) -> None:
    """Write full baseline-subtracted shift arrays and workbooks."""
    report = StageReport(2, "compare", ws.dir.name)
    nodes = [node for node in ws.catalog.all_nodes() if ws.has_cube(node["id"])]
    report.line(
        f"shifting {len(nodes)} nodes against baseline · "
        f"{len(ws.deltas) * ws.n_bins * len(ws.horizons):,} cells per full-bin node"
    )
    warnings = _write_shift_array(
        ws, MilestoneProgress(report, "calculating arrays", len(nodes))
    )
    render_nodes = [node for node in nodes if ws.has_shift_cube(node["id"])]
    written, render_warnings = _render_shift(
        ws, MilestoneProgress(report, "writing spreadsheets", len(render_nodes))
    )
    warnings.extend(render_warnings)
    report.summary(
        f"wrote {len(nodes) - sum('shift skipped' in warning for warning in warnings)} arrays "
        f"+ {written} workbooks → {ws.dir.relative_to(ws.root_dir)}/02_shift"
    )
    report.completed()
    StageReport.warnings(warnings)
