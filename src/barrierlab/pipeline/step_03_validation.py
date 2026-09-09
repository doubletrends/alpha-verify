"""Stage 3: validate every eligible condition bin from stored Stage 2 histories."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib

import numpy as np

from barrierlab.domain import barrier, scoring, validation
from barrierlab.domain.features import is_ohlcv_feature
from barrierlab.infrastructure import artifact_io
from barrierlab.infrastructure.artifacts import feature_from_artifact, market_data_from_artifact
from barrierlab.infrastructure.workspace import BASELINE_NODE, Workspace
from barrierlab.pipeline.reporting import MilestoneProgress, StageReport

N_PATHS = 10_000
SEED = 20260907
RAW_P_THRESHOLD = 0.05
NULL_VERSION = "gaussian-log-ohlc-v1"


def _method() -> dict:
    return {
        "scoring_version": scoring.SCORING_VERSION,
        "measurement_version": barrier.MEASUREMENT_VERSION,
        "observed_source": "stored OHLCV history remeasured in float64",
        "feature_policy": "core features and quantiles recomputed; external values and edges fixed for both roles",
        "null_version": NULL_VERSION,
        "n_paths": N_PATHS, "seed": SEED,
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
        nodes.append({"node": node, "sha256": digest})
    return {"n_bins": ws.n_bins, "nodes": nodes}


def validation_summary_is_current(ws: Workspace, summary: dict) -> bool:
    """A complete summary must match current inputs and simulation settings."""
    if (not summary or not summary.get("complete")
            or summary.get("artifact") != "03_validation"
            or summary.get("method") != _method()):
        return False
    return summary.get("input_fingerprint") == input_fingerprint(ws)


def _history_key(data) -> str:
    digest = hashlib.sha256()
    digest.update(str(tuple(data.columns)).encode())
    digest.update(data.index.asi8.tobytes())
    digest.update(np.ascontiguousarray(data.to_numpy(dtype=float)).tobytes())
    return digest.hexdigest()


def cmd_validation(ws: Workspace) -> None:
    """Validate all available non-baseline nodes without ranking or preselection."""
    report = StageReport(3, "validate", ws.dir.name)
    fingerprint = input_fingerprint(ws)
    missing = [entry["node"]["id"] for entry in fingerprint["nodes"] if entry["sha256"] is None]
    available = [entry["node"] for entry in fingerprint["nodes"] if entry["sha256"] is not None]
    report.line(f"testing all eligible bins from {len(available)} nodes against {N_PATHS:,} null histories")

    # Group references, not full probability cubes; retain only one synthetic
    # ensemble at a time, and never apply one market's null to another history.
    groups = {}
    for node in available:
        cube = artifact_io.load_shift(ws.shift_cube_path(node["id"]))
        data = market_data_from_artifact(cube)
        groups.setdefault(_history_key(data), []).append(node)
    records, skipped = [], []
    progress = MilestoneProgress(report, "validating nodes", len(available))
    for nodes in groups.values():
        simulated_paths = None
        for node in nodes:
            cube = artifact_io.load_shift(ws.shift_cube_path(node["id"]))
            data = market_data_from_artifact(cube)
            fixed = not is_ohlcv_feature(node["feature"])
            # Choose the policy once and pass it unchanged to both roles.
            policy = {
                "features": feature_from_artifact(cube, data.index).to_numpy(float)[None] if fixed else None,
                "edges": cube["edges"] if fixed else None,
                "feature_name": node["feature"], "params": node["params"],
                "deltas": cube["Δs"], "horizons": cube["horizons"], "n_bins": ws.n_bins,
            }
            observed = validation.score_histories(barrier.ohlcv_tensor(data), **policy)
            observed_edges = observed["edges"][0]
            observed_edges = observed_edges[np.isfinite(observed_edges)]
            bin_count = len(observed_edges) + 1
            scores = observed["scores"][0, :bin_count]
            valid = observed["valid"][0, :bin_count]
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
                if simulated_paths is None:
                    simulated_paths = validation.simulated_ohlc_tensor(data, N_PATHS, SEED)
                null = validation.score_histories(simulated_paths, **policy)
                for b in np.flatnonzero(valid):
                    sample = null["scores"][:, b]
                    p_value = float((1 + (sample >= scores[b]).sum()) / (1 + len(sample)))
                    records.append({
                        **identities[b], "bin_score": float(scores[b]),
                        "peak_p": p_value, "cleared": p_value < RAW_P_THRESHOLD,
                        "null_p95": float(np.percentile(sample, 95)),
                        "n_paths": len(sample), "n_valid_null": int(null["valid"][:, b].sum()),
                        "null_scores": sample.tolist(),
                    })
            progress.advance()
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
