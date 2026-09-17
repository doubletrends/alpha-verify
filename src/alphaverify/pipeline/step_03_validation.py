"""Stage 3: validate every eligible condition bin from stored Stage 2 histories."""

from __future__ import annotations

from datetime import datetime, timezone
import math

import numpy as np

from alphaverify.domain import barrier, scoring, tensor_runtime, validation
from alphaverify.domain.features import is_ohlcv_feature
from alphaverify.infrastructure import artifact_io
from alphaverify.infrastructure.artifact_history import (
    feature_from_artifact, market_data_from_artifact, market_history_key,
)
from alphaverify.infrastructure.workspace import BASELINE_NODE, STAGE_DIRECTORIES, Workspace
from alphaverify.pipeline.context import RunContext, materialized_shift
from alphaverify.pipeline.reporting import MilestoneProgress, StageReport

N_NULL_REPLICATES = 1_000
SEED = 20260907
RAW_P_THRESHOLD = 0.05
NULL_VERSION = "gaussian-log-ohlc-v1"


METHOD = {
    "scoring_version": scoring.SCORING_VERSION,
    "measurement_version": barrier.MEASUREMENT_VERSION,
    "observed_source": "stored OHLCV history remeasured in float64",
    "feature_policy": "observed feature bins cached by Stage 1; core null features recomputed; external null bins fixed",
    "null_version": NULL_VERSION,
    "n_null_replicates": N_NULL_REPLICATES, "seed": SEED,
    "threshold": {"raw_p": RAW_P_THRESHOLD},
    "unit": "linearly barrier-weighted probability difference",
    "null": "shared synthetic OHLC histories per identical stored market history; external conditions fixed",
    "invalid_null_bins": "zero score; retained in the full null ensemble",
}


def input_fingerprint(ws: Workspace) -> dict:
    """Certify source bytes, node definitions, and requested quantile count."""
    nodes = []
    for node in sorted(ws.catalog.all_nodes(), key=lambda item: item["id"]):
        if node["id"] == BASELINE_NODE:
            continue
        shift_path, surface_path = ws.shift_cube_path(node["id"]), ws.cube_path(node["id"])
        nodes.append({
            "node": node,
            "sha256": artifact_io.file_sha256(shift_path) if shift_path.exists() else None,
            "source_sha256": artifact_io.file_sha256(surface_path) if surface_path.exists() else None,
        })
    return {"n_bins": ws.n_bins, "nodes": nodes}


def validation_summary_is_current(ws: Workspace, summary: dict) -> bool:
    """A complete summary must match current inputs and simulation settings."""
    if (not summary or not summary.get("complete")
            or summary.get("artifact") != STAGE_DIRECTORIES["validation"]
            or summary.get("method") != METHOD):
        return False
    return summary.get("input_fingerprint") == input_fingerprint(ws)


def cmd_validation(ws: Workspace) -> None:
    """Validate all available non-baseline nodes without ranking or preselection."""
    report = StageReport("validation")
    context = RunContext(ws)
    fingerprint = input_fingerprint(ws)
    missing = [entry["node"]["id"] for entry in fingerprint["nodes"] if entry["sha256"] is None]
    available = [entry["node"] for entry in fingerprint["nodes"] if entry["sha256"] is not None]
    report.line(
        f"testing all eligible bins from {len(available)} nodes against "
        f"{N_NULL_REPLICATES:,} null replicates"
    )

    # Group references, not full probability cubes; retain only one synthetic
    # ensemble at a time, and never apply one market's null to another history.
    groups = {}
    for node in available:
        cube = materialized_shift(ws, node["id"])
        data = market_data_from_artifact(cube)
        groups.setdefault(market_history_key(data), []).append(node)
    records, skipped = [], []
    progress = MilestoneProgress(report, "validating nodes", len(available))
    for nodes in groups.values():
        pending = []
        for node in nodes:
            cube = materialized_shift(ws, node["id"])
            data = market_data_from_artifact(cube)
            outcomes = context.observed_outcomes(data, cube["barriers"], cube["horizons"])
            fixed = not is_ohlcv_feature(node["feature"])
            # Both roles may use the stored condition; the observed role also
            # reuses Stage 1 bins for core features, which the null recomputes.
            cached_condition = fixed or "bin_assignments" in cube
            features = (feature_from_artifact(cube, data.index).to_numpy(float)[None]
                        if cached_condition else None)
            policy = {
                "features": features if fixed else None,
                "bin_edges": cube["bin_edges"] if fixed else None,
                "feature_name": node["feature"], "params": node["params"],
                "barriers": cube["barriers"], "horizons": cube["horizons"],
                "requested_bin_count": ws.n_bins,
            }
            observed = validation.score_histories(
                barrier.ohlcv_tensor(data),
                **{**policy, "features": features,
                   "bin_edges": cube["bin_edges"] if cached_condition else None},
                touch_mask=outcomes["touch_mask"],
                baseline_probability=outcomes["baseline_probability"],
                bin_assignments=(
                    cube.get("bin_assignments") if cached_condition else None
                ),
            )
            observed_edges = observed.bin_edges[0]
            observed_edges = observed_edges[np.isfinite(observed_edges)]
            bin_count = len(observed_edges) + 1
            scores = observed.bin_score[0, :bin_count]
            valid = observed.score_supported[0, :bin_count]
            labels = barrier.bin_labels(observed_edges)
            identities = [
                {"node": node["id"], "family": node["family"],
                 "feature": node["feature"], "params": node["params"],
                 "bin": b, "bin_number": b + 1,
                 "bin_label": labels[b] if b < len(labels) else f"bin {b + 1}"}
                for b in range(len(scores))
            ]
            skipped.extend({**identities[b], "reason": "no eligible cells"}
                           for b in np.flatnonzero(~valid))
            if valid.any():
                pending.append((identities, scores, valid, policy))
            progress.advance()

        if pending:
            simulated_paths = validation.simulated_ohlc_tensor(
                data, N_NULL_REPLICATES, SEED
            )
            batch_size = validation.replicate_batch_size(
                simulated_paths.shape[1], len(pending), len(pending[0][3]["barriers"]),
                tensor_runtime.memory_budget_bytes(),
            )
            nulls = validation.score_histories_many(
                simulated_paths, [item[3] for item in pending], batch_size=batch_size,
                progress=MilestoneProgress(
                    report, "scoring null matrices", math.ceil(len(simulated_paths) / batch_size)
                ),
            )
            for (identities, scores, valid, _), null in zip(pending, nulls):
                for b in np.flatnonzero(valid):
                    sample = null.bin_score[:, b]
                    p_value = float((1 + (sample >= scores[b]).sum()) / (1 + len(sample)))
                    records.append({
                        **identities[b], "bin_score": float(scores[b]),
                        "monte_carlo_p_value": p_value,
                        "cleared": p_value < RAW_P_THRESHOLD,
                        "null_p95": float(np.percentile(sample, 95)),
                        "n_null_replicates": len(sample),
                        "n_supported_null": int(null.score_supported[:, b].sum()),
                        "null_scores": sample.tolist(),
                    })
            del simulated_paths

    if input_fingerprint(ws) != fingerprint:
        raise RuntimeError("Stage 2 inputs changed during validation; rerun validate")
    cleared = [{key: value for key, value in row.items() if key != "null_scores"}
               for row in records if row["cleared"]]
    summary = {
        "workspace": ws.dir.name, "generated": datetime.now(timezone.utc).isoformat(),
        "artifact": STAGE_DIRECTORIES["validation"], "complete": bool(available) and not missing,
        "input_fingerprint": fingerprint, "method": METHOD, "missing_nodes": missing,
        "summary": {"nodes": len(available), "tested": len(records),
                    "skipped": len(skipped), "cleared": len(cleared)},
        "tests": records, "skipped_bins": skipped, "cleared": cleared,
    }
    artifact_io.write_json(ws.validation_summary_path, summary)
    # Deferred so commands that write no figures do not pay matplotlib's import.
    from alphaverify.presentation import bin_figures
    figures = bin_figures.write_bin_figures(
        ws.stage_dir("validation") / "plot",
        records,
        {node: materialized_shift(ws, node) for node in {row["node"] for row in records}},
        workspace=ws.dir.name, horizon_unit=ws.horizon_unit,
        progress=MilestoneProgress(report, "writing bin figures", len(records)),
    )
    report.summary(f"cleared {len(cleared)} of {len(records)} with raw p < {RAW_P_THRESHOLD}; skipped {len(skipped)} unsupported bins")
    if missing:
        report.line(f"incomplete: {len(missing)} nodes lack shift arrays; run measure and compare")
    report.line(f"wrote 1 summary + {len(figures)} figures → {ws.validation_summary_path.parent}")
    report.completed()
