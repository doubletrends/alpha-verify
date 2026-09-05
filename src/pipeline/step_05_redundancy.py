"""Stage 5: map redundancy among nodes cleared by validation."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json

import numpy as np

from domain import bayes, redundancy, shift
from infrastructure.workspace import BASELINE_NODE, Workspace
from pipeline.context import artifact_feature_panel
from pipeline.step_04_validation import validation_summary_is_current
from presentation import workbooks


REDUNDANCY_THRESHOLD = 0.30


def validation_fingerprint(validation: dict) -> str:
    """Hash the Stage 4 decisions that define the Stage 5 input universe."""
    payload = {
        "selection_fingerprint": validation.get("selection_fingerprint"),
        "method": validation.get("method"),
        "cleared": validation.get("cleared"),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def redundancy_artifacts_are_current(
    ws: Workspace,
    validation: dict,
    manifest: dict,
) -> bool:
    """Check the manifest, numerical artifact, workbook, and Stage 4 lineage."""
    if not validation_summary_is_current(ws, validation):
        return False
    if not manifest or not manifest.get("complete"):
        return False
    fingerprint = validation_fingerprint(validation)
    if manifest.get("source_validation_fingerprint") != fingerprint:
        return False
    if not ws.redundancy_array_path.exists() or not ws.redundancy_workbook_path.exists():
        return False
    try:
        artifact = redundancy.load(ws.redundancy_array_path)
    except Exception:
        return False
    names = [str(value) for value in artifact.get("node_ids", [])]
    matrix = artifact.get("conditional_nmi", np.empty((0, 0)))
    return (
        artifact.get("meta", {}).get("source_validation_fingerprint") == fingerprint
        and matrix.shape == (len(names), len(names))
        and names == [row.get("node") for row in manifest.get("nodes", [])]
        and float(np.asarray(artifact.get("threshold", np.nan)))
        == float(manifest.get("method", {}).get("threshold", np.nan))
    )


def _cluster_rows(
    groups: list[list[int]],
    names: list[str],
    families: dict[str, str],
    relevance: np.ndarray,
    similarity: np.ndarray,
) -> tuple[list[dict], np.ndarray, np.ndarray, np.ndarray]:
    cluster_ids = np.zeros(len(names), dtype=np.int32)
    representatives = np.zeros(len(names), dtype=bool)
    representative_indices = []
    rows = []
    for cluster_id, members in enumerate(groups, 1):
        best = max(members, key=lambda index: relevance[index])
        representative_indices.append(best)
        representatives[best] = True
        cluster_ids[members] = cluster_id
        internal = [
            float(similarity[left, right])
            for offset, left in enumerate(members)
            for right in members[offset + 1:]
        ]
        rows.append({
            "cluster": cluster_id,
            "representative": names[best],
            "members": [names[index] for index in members],
            "families": sorted({families[names[index]] for index in members}),
            "size": len(members),
            "representative_information": float(relevance[best]),
            "mean_internal_nmi": float(np.mean(internal)) if internal else 0.0,
            "max_internal_nmi": max(internal, default=0.0),
        })
    return rows, cluster_ids, representatives, np.asarray(representative_indices, dtype=np.int32)


def cmd_redundancy(ws: Workspace) -> None:
    """Write numerical, workbook, and manifest views of full-history redundancy."""
    validation = ws.read_json(ws.validation_summary_path)
    if not validation_summary_is_current(ws, validation):
        print("No current complete validation summary - run validation first.")
        return
    cleared = validation.get("cleared", [])
    node_ids = {row["node"] for row in cleared}
    if not node_ids:
        print("No cleared nodes - run validation first.")
        return

    baseline_path = ws.shift_cube_path("_base", BASELINE_NODE)
    if not baseline_path.exists():
        print("No baseline shift array - run shift first.")
        return
    delta, horizon = ws.composition_target(shift.load(baseline_path)["base"])
    try:
        data, features, families = artifact_feature_panel(ws)
    except ValueError as error:
        print(f"Cannot build redundancy map: {error}")
        return

    names = [name for name in features if name in node_ids]
    if len(names) < 2:
        print("Need at least two validation-cleared nodes for a redundancy map.")
        return
    X = np.vstack([features[name].to_numpy(float) for name in names])
    y = bayes.touch_label(data, delta, horizon)
    ok = np.isfinite(y) & np.isfinite(X).all(axis=0)
    X, y = X[:, ok], y[ok].astype(int)
    cuts = np.linspace(0, 1, ws.n_bins + 1)[1:-1]
    edges = [np.unique(np.quantile(X[index], cuts)) for index in range(len(names))]
    n_bins = np.array([len(edge) + 1 for edge in edges])
    bins = np.vstack([np.searchsorted(edges[index], X[index]) for index in range(len(names))])
    model = bayes.fit(y, bins, n_bins)
    relevance = bayes.information_scores(model, bins, n_bins)
    similarity = redundancy.conditional_nmi_matrix(bins, y)
    groups = redundancy.clusters(similarity, REDUNDANCY_THRESHOLD)
    cluster_rows, cluster_ids, representatives, representative_indices = _cluster_rows(
        groups, names, families, relevance, similarity
    )

    pairs = [
        {
            "left": names[left],
            "right": names[right],
            "conditional_nmi": float(similarity[left, right]),
            "above_threshold": bool(similarity[left, right] >= REDUNDANCY_THRESHOLD),
            "same_cluster": bool(cluster_ids[left] == cluster_ids[right]),
        }
        for left in range(len(names)) for right in range(left + 1, len(names))
    ]
    pairs.sort(key=lambda row: -row["conditional_nmi"])
    validation_by_node = {row["node"]: row for row in cleared}
    node_rows = [
        {
            "node": name,
            "family": families[name],
            "information": float(relevance[index]),
            "cluster": int(cluster_ids[index]),
            "representative": bool(representatives[index]),
        }
        for index, name in enumerate(names)
    ]

    generated = datetime.now(timezone.utc).isoformat()
    source_fingerprint = validation_fingerprint(validation)
    meta = {
        "artifact": "05_redundancy",
        "workspace": ws.dir.name,
        "generated": generated,
        "source_validation_fingerprint": source_fingerprint,
        "target": {"Δ": delta, "horizon": horizon, "unit": ws.horizon_unit},
        "method": {
            "similarity": "normalized I(bin_i; bin_j | touch outcome)",
            "threshold": REDUNDANCY_THRESHOLD,
            "clustering": "connected components at or above threshold",
            "representative": "greatest individual target information within cluster",
        },
        "n_observations": int(len(y)),
        "outcome_rate": float(y.mean()),
    }
    files = {
        "array": str(ws.redundancy_array_path.relative_to(ws.dir)).replace("\\", "/"),
        "workbook": str(ws.redundancy_workbook_path.relative_to(ws.dir)).replace("\\", "/"),
    }
    pending = {
        "artifact": "05_redundancy",
        "workspace": ws.dir.name,
        "generated": generated,
        "complete": False,
        "source_validation_fingerprint": source_fingerprint,
        "target": meta["target"],
        "method": meta["method"],
        "files": files,
        "summary": {},
        "nodes": [],
        "clusters": [],
    }
    ws.write_json(ws.redundancy_path, pending)

    redundancy.save(ws.redundancy_array_path, {
        "node_ids": np.asarray(names, dtype=str),
        "families": np.asarray([families[name] for name in names], dtype=str),
        "information": relevance.astype(np.float64),
        "conditional_nmi": similarity.astype(np.float64),
        "cluster_id": cluster_ids,
        "representative": representatives,
        "representative_index": representative_indices,
        "target_delta": np.array(delta, dtype=np.float64),
        "target_horizon": np.array(horizon, dtype=np.int32),
        "threshold": np.array(REDUNDANCY_THRESHOLD, dtype=np.float64),
        "n_observations": np.array(len(y), dtype=np.int64),
        "outcome_rate": np.array(y.mean(), dtype=np.float64),
    }, meta)
    try:
        workbooks.write_redundancy_xlsx(
            ws.redundancy_workbook_path,
            names,
            [families[name] for name in names],
            relevance,
            similarity,
            cluster_ids,
            representatives,
            cluster_rows,
            pairs,
            validation_by_node,
            meta,
        )
    except PermissionError:
        print("Redundancy workbook is locked; close it in Excel and run redundancy again.")
        return

    above_threshold = sum(row["above_threshold"] for row in pairs)
    manifest = {
        **pending,
        "complete": True,
        "summary": {
            "nodes": len(names),
            "clusters": len(groups),
            "pairs": len(pairs),
            "pairs_above_threshold": above_threshold,
            "n_observations": len(y),
            "outcome_rate": float(y.mean()),
        },
        "nodes": node_rows,
        "clusters": cluster_rows,
    }
    ws.write_json(ws.redundancy_path, manifest)

    print(f"\n=== 5. Redundancy [{ws.dir.name}] ===")
    print(f"  {len(names)} validation-cleared nodes → {len(groups)} conditional-dependence clusters")
    for group in cluster_rows:
        members = ", ".join(group["members"])
        print(f"  {group['cluster']:>2}. {group['representative']:<26} {members}")
    print(f"\n  wrote {ws.redundancy_array_path.relative_to(ws.root_dir)}")
    print(f"  wrote {ws.redundancy_workbook_path.relative_to(ws.root_dir)}")
    print(f"  wrote {ws.redundancy_path.relative_to(ws.root_dir)}")
