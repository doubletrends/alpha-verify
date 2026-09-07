"""Stage 1: full barrier-touch surface measurement and workbooks."""

from __future__ import annotations

from datetime import datetime, timezone

from barrierlab.domain import barrier
from barrierlab.infrastructure import artifact_io
from barrierlab.infrastructure.workspace import Workspace
from barrierlab.pipeline.context import RunContext
from barrierlab.pipeline.reporting import MilestoneProgress, StageReport
from barrierlab.presentation import workbooks


def _build_cube(context: RunContext, node: dict) -> tuple[tuple[int, ...], int]:
    workspace = context.workspace
    data, feature = context.node_feature(node)
    edges = barrier.bin_edges(feature, workspace.n_bins)
    cube = barrier.touch_tensor(
        data,
        feature,
        workspace.horizons,
        workspace.deltas,
        edges,
        excursions=context.forward_excursions(node["data"], data, int(workspace.horizons.max())),
    )
    cube["index"] = data.index.astype(str).to_numpy()
    cube["feature_values"] = feature.reindex(data.index).to_numpy(float)
    for column in ("open", "high", "low", "close", "volume"):
        if column in data:
            cube[column] = data[column].to_numpy(float)

    artifact_io.save_surface(cube, workspace.cube_path(node["id"]), {
        "node": node["id"],
        "family": node["family"],
        "feature": node["feature"],
        "params": node["params"],
        "workspace": workspace.dir.name,
        "bin_labels": barrier.bin_labels(edges, feature),
        "grid": "full",
        "generated": datetime.now(timezone.utc).isoformat(),
    })
    return cube["prob"].shape, int(cube["n_obs"][0])


def _write_surface_arrays(
    workspace: Workspace, verbose: bool, progress: MilestoneProgress
) -> list[str]:
    nodes = workspace.catalog.all_nodes()
    context = RunContext(workspace)
    horizons = workspace.horizons
    skipped = {}
    for node in nodes:
        try:
            shape, n_obs = _build_cube(context, node)
            if verbose:
                print(f"  {node['id']:<26} {shape}  n={n_obs}")
        except Exception as error:
            skipped[node["id"]] = str(error)
        finally:
            progress.advance()
    return [f"surface skipped {node}: {error}" for node, error in skipped.items()]


def _render_surface(
    workspace: Workspace, progress: MilestoneProgress
) -> tuple[int, list[str]]:
    nodes = [
        node for node in workspace.catalog.all_nodes()
        if workspace.has_cube(node["id"])
    ]
    if not nodes:
        return 0, ["no full surface arrays available; run measure first"]
    written = 0
    warnings = []
    for node in nodes:
        cube = artifact_io.load_surface(workspace.cube_path(node["id"]))
        try:
            workbooks.write_barrier_xlsx(
                cube,
                workspace.surface_path(node["id"]),
                node["id"],
                node["feature"],
                node["params"],
                workspace.horizon_unit,
            )
        except PermissionError:
            warnings.append(f"workbook locked for {node['id']}; close it in Excel and re-run")
        else:
            written += 1
        finally:
            progress.advance()
    return written, warnings


def cmd_surface(workspace: Workspace, *, verbose: bool = False) -> None:
    """Write full surface arrays and their workbook views."""
    report = StageReport(1, "measure", workspace.dir.name)
    nodes = workspace.catalog.all_nodes()
    report.line(
        f"measuring {len(nodes)} nodes · {len(workspace.deltas)} Δ × "
        f"{workspace.n_bins} bins × {len(workspace.horizons)} horizons"
    )
    warnings = _write_surface_arrays(
        workspace, verbose, MilestoneProgress(report, "calculating arrays", len(nodes))
    )
    render_nodes = [node for node in nodes if workspace.has_cube(node["id"])]
    written, render_warnings = _render_surface(
        workspace, MilestoneProgress(report, "writing spreadsheets", len(render_nodes))
    )
    warnings.extend(render_warnings)
    report.summary(
        f"wrote {len(nodes) - sum('surface skipped' in warning for warning in warnings)} arrays "
        f"+ {written} workbooks → {workspace.dir.relative_to(workspace.root_dir)}/01_surface"
    )
    report.completed()
    StageReport.warnings(warnings)
