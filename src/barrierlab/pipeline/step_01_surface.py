"""Stage 1: full barrier-touch surface measurement and workbooks."""

from __future__ import annotations

from datetime import datetime, timezone

from barrierlab.domain import barrier
from barrierlab.infrastructure import artifact_io
from barrierlab.infrastructure.workspace import Workspace
from barrierlab.pipeline.context import RunContext
from barrierlab.presentation import workbooks


def _build_cube(context: RunContext, node: dict) -> None:
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

    artifact_io.save_surface(cube, workspace.cube_path(node["family"], node["id"]), {
        "node": node["id"],
        "family": node["family"],
        "feature": node["feature"],
        "params": node["params"],
        "workspace": workspace.dir.name,
        "bin_labels": barrier.bin_labels(edges, feature),
        "grid": "full",
        "generated": datetime.now(timezone.utc).isoformat(),
    })
    print(f"  {node['id']:<26} {cube['prob'].shape}  n={int(cube['n_obs'][0])}")


def _write_surface_arrays(workspace: Workspace) -> None:
    nodes = workspace.catalog.all_nodes()
    context = RunContext(workspace)
    horizons = workspace.horizons
    print(f"\n=== 1. 01_surface arrays [{workspace.dir.name}] - {len(nodes)} nodes ===")
    print(
        f"  {len(workspace.deltas)} Delta x {workspace.n_bins} bins x {len(horizons)} horizons "
        f"(+{horizons[0]}{workspace.horizon_unit}..+{horizons[-1]}{workspace.horizon_unit})"
    )
    print("  value = P(touch Delta in t | bin); intraday high/low\n")

    skipped = {}
    for node in nodes:
        try:
            _build_cube(context, node)
        except Exception as error:
            skipped[node["id"]] = str(error)
            print(f"  {node['id']:<26} [skip] {error}")
    suffix = f"   ({len(skipped)} skipped)" if skipped else ""
    print(f"\n  wrote to {workspace.dir.relative_to(workspace.root_dir)}/01_surface/{suffix}")


def _render_surface(workspace: Workspace) -> None:
    nodes = [
        node for node in workspace.catalog.all_nodes()
        if workspace.has_cube(node["family"], node["id"])
    ]
    if not nodes:
        print("No full arrays to render - run surface first.")
        return

    print(f"\n=== 1. 01_surface workbooks [{workspace.dir.name}] - {len(nodes)} nodes ===")
    print(f"  {workspace.n_bins} tabs per node, one per condition bin\n")
    written = 0
    for node in nodes:
        cube = artifact_io.load_surface(workspace.cube_path(node["family"], node["id"]))
        try:
            workbooks.write_barrier_xlsx(
                cube,
                workspace.surface_path(node["family"], node["id"]),
                node["id"],
                node["feature"],
                node["params"],
                workspace.horizon_unit,
            )
        except PermissionError:
            print(f"  {node['id']:<26} [locked] close it in Excel and re-run")
            continue
        written += 1
    print(f"  wrote {written} workbooks under {workspace.dir.relative_to(workspace.root_dir)}/01_surface/")


def cmd_surface(workspace: Workspace) -> None:
    """Write full surface arrays and their workbook views."""
    _write_surface_arrays(workspace)
    _render_surface(workspace)
