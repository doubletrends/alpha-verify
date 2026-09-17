"""Stage 2: baseline-subtracted shift artifacts and workbooks."""

from __future__ import annotations

from alphaverify.domain import shift
from alphaverify.infrastructure import artifact_io
from alphaverify.infrastructure.workspace import BASELINE_NODE, Workspace
from alphaverify.pipeline.context import baseline_surface, materialized_shift
from alphaverify.pipeline.reporting import MilestoneProgress, StageReport
from alphaverify.presentation import workbooks


def _write_shift_array(
    ws: Workspace, nodes: list[dict], progress: MilestoneProgress
) -> tuple[int, list[str]]:
    if not nodes:
        return 0, ["no full surface arrays available; run measure first"]

    baseline = baseline_surface(ws)
    if baseline is None:
        return 0, ["no baseline surface array available; run measure first"]

    nodes = [node for node in nodes if node["id"] == BASELINE_NODE] + [
        node for node in nodes if node["id"] != BASELINE_NODE
    ]
    written = 0
    warnings = []
    for node in nodes:
        try:
            source = ws.cube_path(node["id"])
            full = artifact_io.load_surface(source)
            artifact_io.save_shift(
                shift.from_cube(full, baseline),
                ws.shift_cube_path(node["id"]),
                {**full["meta"], "grid": "shift", "value": "probability_shift",
                 "source_artifact": str(source.relative_to(ws.dir)),
                 "source_sha256": artifact_io.file_sha256(source),
                 "baseline_artifact": str(ws.baseline_cube.relative_to(ws.dir)),
                 "baseline_sha256": artifact_io.file_sha256(ws.baseline_cube)},
            )
        except Exception as error:
            warnings.append(f"shift skipped {node['id']}: {error}")
        else:
            written += 1
        finally:
            progress.advance()
    return written, warnings


def _render_shift(
    ws: Workspace, nodes: list[dict], progress: MilestoneProgress
) -> tuple[int, list[str]]:
    if not nodes:
        return 0, ["no shift arrays available; run compare first"]
    written = 0
    warnings = []
    for node in nodes:
        cube = materialized_shift(ws, node["id"])
        try:
            workbooks.write_shift_xlsx(
                cube,
                ws.shift_surface_path(node["id"]),
                node["id"],
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
    report = StageReport("shift")
    nodes = [node for node in ws.catalog.all_nodes() if ws.has_cube(node["id"])]
    report.line(
        f"shifting {len(nodes)} nodes against baseline · "
        f"{len(ws.barriers) * ws.n_bins * len(ws.horizons):,} cells per full-bin node"
    )
    arrays, warnings = _write_shift_array(
        ws, nodes, MilestoneProgress(report, "calculating arrays", len(nodes))
    )
    render_nodes = [node for node in nodes if ws.has_shift_cube(node["id"])]
    workbooks_written, render_warnings = _render_shift(
        ws, render_nodes, MilestoneProgress(report, "writing spreadsheets", len(render_nodes)),
    )
    warnings.extend(render_warnings)
    report.summary(
        f"wrote {arrays} arrays + {workbooks_written} workbooks → "
        f"{ws.stage_dir('shift').relative_to(ws.root_dir)}"
    )
    report.completed()
    StageReport.warnings(warnings)
