"""Stage 3: validate every eligible condition bin from stored Stage 2 histories."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import math

import numpy as np

from barrierlab.domain import barrier, scoring, validation
from barrierlab.domain.features import is_ohlcv_feature
from barrierlab.infrastructure import artifact_io
from barrierlab.infrastructure.artifacts import (
    feature_from_artifact, market_data_from_artifact, market_history_key,
)
from barrierlab.infrastructure.workspace import BASELINE_NODE, Workspace
from barrierlab.pipeline.context import RunContext, materialized_shift
from barrierlab.pipeline.reporting import MilestoneProgress, StageReport

N_NULL_REPLICATES = 1_000
SEED = 20260907
RAW_P_THRESHOLD = 0.05
NULL_VERSION = "gaussian-log-ohlc-v1"


def _method() -> dict:
    return {
        "scoring_version": scoring.SCORING_VERSION,
        "measurement_version": barrier.MEASUREMENT_VERSION,
        "observed_source": "stored OHLCV history remeasured in float64",
        "feature_policy": "observed feature bins cached by Stage 1; core null features recomputed; external null bins fixed",
        "null_version": NULL_VERSION,
        "n_null_replicates": N_NULL_REPLICATES, "seed": SEED,
        "threshold": {"raw_p": RAW_P_THRESHOLD},
        "unit": "linearly barrier-weighted percentage points",
        "null": "shared synthetic OHLC histories per identical stored market history; external conditions fixed",
        "invalid_null_bins": "zero score; retained in the full null ensemble",
    }


def input_fingerprint(ws: Workspace) -> dict:
    """Certify source bytes, node definitions, and requested quantile count."""
    nodes = []
    for node in sorted(ws.catalog.all_nodes(), key=lambda item: item["id"]):
        if node["id"] == BASELINE_NODE:
            continue
        path = ws.shift_cube_path(node["id"])
        digest = None
        if path.exists():
            with path.open("rb") as source:
                digest = hashlib.file_digest(source, "sha256").hexdigest()
        surface_digest = None
        surface_path = ws.cube_path(node["id"])
        if surface_path.exists():
            with surface_path.open("rb") as source:
                surface_digest = hashlib.file_digest(source, "sha256").hexdigest()
        nodes.append({"node": node, "sha256": digest, "source_sha256": surface_digest})
    return {"n_bins": ws.n_bins, "nodes": nodes}


def validation_summary_is_current(ws: Workspace, summary: dict) -> bool:
    """A complete summary must match current inputs and simulation settings."""
    if (not summary or not summary.get("complete")
            or summary.get("artifact") != "03_validation"
            or summary.get("method") != _method()):
        return False
    return summary.get("input_fingerprint") == input_fingerprint(ws)


def _input_cube(ws: Workspace, node_id: str) -> dict:
    """Read legacy rich shifts or hydrate a thin shift from its Stage 1 source."""
    return materialized_shift(ws, node_id)


def cmd_validation(ws: Workspace) -> None:
    """Validate all available non-baseline nodes without ranking or preselection."""
    report = StageReport(3, "validate", ws.dir.name)
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
        cube = _input_cube(ws, node["id"])
        data = market_data_from_artifact(cube)
        groups.setdefault(market_history_key(data), []).append(node)
    records, skipped = [], []
    progress = MilestoneProgress(report, "validating nodes", len(available))
    for nodes in groups.values():
        pending = []
        for node in nodes:
            cube = _input_cube(ws, node["id"])
            data = market_data_from_artifact(cube)
            outcomes = context.observed_outcomes(data, cube["barriers"], cube["horizons"])
            fixed = not is_ohlcv_feature(node["feature"])
            # Choose the policy once and pass it unchanged to both roles.
            policy = {
                "features": feature_from_artifact(cube, data.index).to_numpy(float)[None] if fixed else None,
                "bin_edges": cube["bin_edges"] if fixed else None,
                "feature_name": node["feature"], "params": node["params"],
                "barriers": cube["barriers"], "horizons": cube["horizons"],
                "requested_bin_count": ws.n_bins,
            }
            cached_condition = fixed or "bin_assignments" in cube
            observed = validation.score_histories(
                barrier.ohlcv_tensor(data),
                features=(feature_from_artifact(cube, data.index).to_numpy(float)[None]
                          if cached_condition else None),
                bin_edges=cube["bin_edges"] if cached_condition else None,
                feature_name=node["feature"], params=node["params"],
                barriers=cube["barriers"], horizons=cube["horizons"],
                requested_bin_count=ws.n_bins,
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
            null_steps = math.ceil(
                len(simulated_paths) / validation.REPLICATE_BATCH_SIZE
            )
            nulls = validation.score_histories_many(
                simulated_paths, [item[3] for item in pending],
                progress=MilestoneProgress(report, "scoring null matrices", null_steps),
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
        "artifact": "03_validation", "complete": bool(available) and not missing,
        "input_fingerprint": fingerprint, "method": _method(), "missing_nodes": missing,
        "summary": {"nodes": len(available), "tested": len(records),
                    "skipped": len(skipped), "cleared": len(cleared)},
        "tests": records, "skipped_bins": skipped, "cleared": cleared,
    }
    ws.write_json(ws.validation_summary_path, summary)
    from barrierlab.presentation.validation_plots import write_bin_score_null_histograms
    plots = write_bin_score_null_histograms(
        ws, summary, MilestoneProgress(report, "writing plots", len(records)),
    )
    report.summary(f"cleared {len(cleared)} of {len(records)} with raw p < {RAW_P_THRESHOLD}; skipped {len(skipped)} unsupported bins")
    if missing:
        report.line(f"incomplete: {len(missing)} nodes lack shift arrays; run measure and compare")
    report.line(f"wrote 1 summary + {len(plots)} plots → {ws.validation_summary_path.parent}")
    report.completed()
