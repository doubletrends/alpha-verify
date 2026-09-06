"""Stage 5: map redundancy among nodes cleared by validation."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json

import numpy as np
import pandas as pd

from barrierlab.domain import bayes, redundancy
from barrierlab.infrastructure import artifact_io
from barrierlab.infrastructure.workspace import Workspace
from barrierlab.pipeline.step_04_validation import validation_summary_is_current
from barrierlab.presentation import workbooks


REDUNDANCY_THRESHOLD = 0.30


def validated_feature_panel(ws: Workspace, validation: dict) -> tuple[pd.DataFrame, dict, dict, dict]:
    """Rebuild Stage 5 inputs exclusively from Stage 4's carried-forward bundle."""
    if not ws.validated_bundle_path.exists():
        raise ValueError("missing validated bundle - run validation first")
    bundle = artifact_io.load_validated_bundle(ws.validated_bundle_path)
    if bundle.get("meta", {}).get("selection_fingerprint") != validation.get("selection_fingerprint"):
        raise ValueError("validated bundle does not match validation summary - run validation first")
    required = {"index", "high", "low", "close", "base", "Δs", "horizons", "node_ids", "families"}
    if required - set(bundle):
        raise ValueError("validated bundle is incomplete - run validation first")
    data = pd.DataFrame({
        "high": bundle["high"].astype(float),
        "low": bundle["low"].astype(float),
        "close": bundle["close"].astype(float),
    }, index=pd.to_datetime(bundle["index"]))
    ids = [str(value) for value in bundle["node_ids"]]
    families = {node_id: str(bundle["families"][index]) for index, node_id in enumerate(ids)}
    features = {
        node_id: pd.Series(bundle[f"feature_{index}"].astype(float), index=data.index)
        for index, node_id in enumerate(ids)
        if f"feature_{index}" in bundle
    }
    return data, features, families, bundle


def fold_representative_plan(
    data: pd.DataFrame, features: dict, names: list[str], delta: float,
    horizon: int, n_bins: int, folds: int,
) -> list[list[str]]:
    """Choose one redundancy-cluster representative per Stage 6 training fold."""
    X = np.vstack([features[name].to_numpy(float) for name in names])
    y_all = bayes.touch_label(data, delta, horizon)
    ok = np.isfinite(y_all) & np.isfinite(X).all(axis=0)
    X, y_all = X[:, ok], y_all[ok].astype(int)
    bounds = np.linspace(int(len(y_all) * 0.5), len(y_all), folds + 1).astype(int)
    cuts = np.linspace(0, 1, n_bins + 1)[1:-1]
    plans: list[list[str]] = []
    for fold in range(folds):
        train_end = int(bounds[fold]) - horizon
        if train_end < 200:
            raise ValueError("not enough training history for Stage 6 representative plan")
        X_train, y_train = X[:, :train_end], y_all[:train_end]
        edges = [np.unique(np.quantile(X_train[index], cuts)) for index in range(len(names))]
        bins = np.vstack([
            np.searchsorted(edges[index], X_train[index]) for index in range(len(names))
        ])
        model = bayes.fit(y_train, bins, np.array([len(edge) + 1 for edge in edges]))
        relevance = bayes.information_scores(model, bins, np.array([len(edge) + 1 for edge in edges]))
        groups = redundancy.clusters(redundancy.conditional_nmi_matrix(bins, y_train), REDUNDANCY_THRESHOLD)
        plans.append(sorted(names[max(group, key=lambda index: relevance[index])] for group in groups))
    return plans


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
        artifact = artifact_io.load_redundancy(ws.redundancy_array_path)
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

    try:
        data, features, families, bundle = validated_feature_panel(ws, validation)
    except ValueError as error:
        print(f"Cannot build Stage 5 handoff: {error}")
        return
    delta, horizon = ws.composition_target(bundle["base"])

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
    representative_names = [names[index] for index in representative_indices]
    fold_plans = fold_representative_plan(
        data, features, names, delta, horizon, ws.n_bins, ws.composition_folds
    )
    current = bayes.current_weighted_forecast(
        data, {name: features[name] for name in representative_names}, delta, horizon,
        n_bins=ws.n_bins, top_k=None, equal_weight=True,
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

    artifact_io.save_redundancy(ws.redundancy_array_path, {
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
        "representative_node_ids": np.asarray(representative_names, dtype=str),
    }, meta)
    composition_arrays = {
        "index": np.asarray(data.index.astype(str), dtype=str),
        "high": data["high"].to_numpy(float),
        "low": data["low"].to_numpy(float),
        "close": data["close"].to_numpy(float),
        "base": np.asarray(bundle["base"], dtype=float),
        "Δs": np.asarray(bundle["Δs"], dtype=float),
        "horizons": np.asarray(bundle["horizons"], dtype=int),
        "node_ids": np.asarray(names, dtype=str),
        "families": np.asarray([families[name] for name in names], dtype=str),
        "representative_node_ids": np.asarray(representative_names, dtype=str),
        "fold_representatives_json": np.asarray(json.dumps(fold_plans, separators=(",", ":"))),
        "validation_json": np.asarray(json.dumps(validation, separators=(",", ":"))),
    }
    selected_ids = [str(value) for value in bundle["node_ids"]]
    selected_indexes = {name: index for index, name in enumerate(selected_ids)}
    for index, name in enumerate(names):
        source_index = selected_indexes[name]
        composition_arrays[f"feature_{index}"] = features[name].to_numpy(float)
        composition_arrays[f"shift_{index}"] = np.asarray(bundle[f"shift_{source_index}"], dtype=float)
        composition_arrays[f"edges_{index}"] = np.asarray(bundle[f"edges_{source_index}"], dtype=float)
    artifact_io.save_composition_inputs(ws.composition_inputs_path, composition_arrays, {
        **meta,
        "artifact": "05_redundancy/composition_inputs",
        "source_validation_fingerprint": source_fingerprint,
        "weight_target": {"Δ": delta, "horizon": horizon, "unit": ws.horizon_unit},
    })
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
        "equal_weight_bayes": {
            "target": {"Δ": delta, "horizon": horizon, "unit": ws.horizon_unit},
            "as_of": current["as_of"],
            "representatives": representative_names,
            "fold_representatives": fold_plans,
            "node_weight": 1.0,
            "note": "every retained representative contributes one equal Naive-Bayes log-odds term",
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
    print(f"  wrote {ws.composition_inputs_path.relative_to(ws.root_dir)}")
    print(f"  wrote {ws.redundancy_workbook_path.relative_to(ws.root_dir)}")
    print(f"  wrote {ws.redundancy_path.relative_to(ws.root_dir)}")
