"""Stage 1: full barrier-touch surface measurement and workbooks."""

from __future__ import annotations

from datetime import datetime, timezone

from alphaverify.domain import barrier
from alphaverify.infrastructure import artifact_io
from alphaverify.infrastructure.workspace import Workspace
from alphaverify.pipeline.context import RunContext
from alphaverify.pipeline.reporting import MilestoneProgress, StageReport
from alphaverify.presentation import workbooks


def _build_cube(context: RunContext, node: dict) -> None:
    workspace = context.workspace
    data, feature = context.node_feature(node)
    edges = barrier.bin_edges(feature, workspace.n_bins)
    outcomes = context.observed_outcomes(data)
    cube = barrier.touch_tensor(
        data,
        feature,
        workspace.horizons,
        workspace.barriers,
        edges,
        excursions=(outcomes["downside_excursion"], outcomes["upside_excursion"]),
        touch_mask=outcomes["touch_mask"],
        baseline_probability=outcomes["baseline_probability"],
    )
    cube["index"] = data.index.astype(str).to_numpy()
    cube["feature_values"] = feature.reindex(data.index).to_numpy(float)
    for column in ("open", "high", "low", "close", "volume"):
        cube[column] = data[column].to_numpy(float)

    artifact_io.save_surface(cube, workspace.cube_path(node["id"]), {
        "node": node["id"],
        "family": node["family"],
        "feature": node["feature"],
        "params": node["params"],
        "workspace": workspace.dir.name,
        "data_provenance": data.attrs.get("provenance", {}),
        "bin_labels": barrier.bin_labels(edges),
        "grid": "full",
        "generated": datetime.now(timezone.utc).isoformat(),
    })


def _write_surface_arrays(
    workspace: Workspace, nodes: list[dict], progress: MilestoneProgress
) -> tuple[int, list[str]]:
    context = RunContext(workspace)
    written = 0
    warnings = []
    for node in nodes:
        try:
            _build_cube(context, node)
        except Exception as error:
            warnings.append(f"surface skipped {node['id']}: {error}")
        else:
            written += 1
        finally:
            progress.advance()
    return written, warnings


def _render_surface(
    workspace: Workspace, nodes: list[dict], progress: MilestoneProgress
) -> tuple[int, list[str]]:
    if not nodes:
        return 0, ["no full surface arrays available; run measure first"]
    written = 0
    warnings = []
    for node in nodes:
        cube = artifact_io.load_surface(workspace.cube_path(node["id"]))
        try:
            workbooks.write_surface_xlsx(
                cube,
                workspace.surface_path(node["id"]),
                node["id"],
                workspace.horizon_unit,
            )
        except PermissionError:
            warnings.append(f"workbook locked for {node['id']}; close it in Excel and re-run")
        else:
            written += 1
        finally:
            progress.advance()
    return written, warnings


def cmd_surface(workspace: Workspace) -> None:
    """Write full surface arrays and their workbook views."""
    report = StageReport("surface")
    nodes = workspace.catalog.all_nodes()
    report.line(
        f"measuring {len(nodes)} nodes · {len(workspace.barriers)} barriers × "
        f"{workspace.n_bins} bins × {len(workspace.horizons)} horizons"
    )
    arrays, warnings = _write_surface_arrays(
        workspace, nodes, MilestoneProgress(report, "calculating arrays", len(nodes))
    )
    render_nodes = [node for node in nodes if workspace.has_cube(node["id"])]
    workbooks_written, render_warnings = _render_surface(
        workspace, render_nodes,
        MilestoneProgress(report, "writing spreadsheets", len(render_nodes)),
    )
    warnings.extend(render_warnings)
    report.summary(
        f"wrote {arrays} arrays + {workbooks_written} workbooks → "
        f"{workspace.stage_dir('surface').relative_to(workspace.root_dir)}"
    )
    report.completed()
    StageReport.warnings(warnings)
