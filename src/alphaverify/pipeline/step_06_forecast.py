"""Stage 6: combine the cleared conditions active on the last stored bar into one forecast."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from alphaverify.domain import combination, shift
from alphaverify.domain.barrier import MIN_BIN_N
from alphaverify.infrastructure import artifact_io
from alphaverify.infrastructure.artifact_history import (
    feature_bins_from_artifact, market_data_from_artifact,
)
from alphaverify.infrastructure.workspace import Workspace
from alphaverify.pipeline.context import RunContext, materialized_shift
from alphaverify.pipeline.reporting import StageReport
from alphaverify.pipeline.status import no_strong_cell_line, strongest_cell_line
from alphaverify.pipeline.step_05_summary import node_summary_is_current
from alphaverify.presentation import workbooks

METHOD = {
    "active": "the last stored bar's feature value falls in a cleared bin under stored Stage 1 edges",
    "family_rule": "at most one active bin per family: best rank, then node and bin",
    "combination": "naive Bayes per cell: logit baseline + sum(logit condition - logit baseline); "
                   "conditions treated as independent given the outcome",
    "smoothing": "(hits + 1) / (n + 2) before log-odds",
    "min_bin_n": MIN_BIN_N,
    "joint": "historical touch rate on bars where every contributing condition held",
}


def forecast_is_current(ws: Workspace, forecast: dict) -> bool:
    """Require a complete forecast from the current summary bytes and this method."""
    summary = artifact_io.read_json(ws.node_summary_path)
    if (not forecast or not forecast.get("complete")
            or forecast.get("artifact") != "06_forecast"
            or forecast.get("method") != METHOD
            or not node_summary_is_current(ws, summary)):
        return False
    return forecast.get("summary_sha256") == artifact_io.file_sha256(ws.node_summary_path)


def active_conditions(summary: dict, cubes: dict[str, dict], baseline_index) -> list[dict]:
    """Cleared bins holding on the last stored bar, marked used under the one-per-family rule."""
    active = []
    for node in summary["nodes"]:
        cube = cubes[node["node"]]
        same_history = np.array_equal(cube["index"], baseline_index)
        latest_bin = int(feature_bins_from_artifact(cube)[-1])
        for item in node["bins"]:
            if item["bin"] != latest_bin:
                continue
            active.append({
                "family": node["family"], "node": node["node"],
                "bin": item["bin"], "bin_number": item["bin_number"],
                "bin_count": len(cube["bin_edges"]) + 1, "bin_label": item["bin_label"],
                "latest_value": float(cube["feature_values"][-1]),
                "rank": item["rank"], "monte_carlo_p_value": item["monte_carlo_p_value"],
                "q_value": item["q_value"], "used": False,
                "note": "" if same_history else "stored history differs from the baseline",
            })
    active.sort(key=lambda row: (row["rank"], row["node"], row["bin"]))
    chosen: dict[str, dict] = {}
    for row in active:
        if row["note"]:
            continue
        if row["family"] in chosen:
            winner = chosen[row["family"]]
            row["note"] = f"same family as {winner['node']} bin {winner['bin_number']} (rank {winner['rank']})"
        else:
            chosen[row["family"]] = row
            row["used"] = True
    return active


def _json_surface(values: np.ndarray) -> list:
    return [[None if not np.isfinite(value) else float(value) for value in row] for row in values]


def cmd_forecast(ws: Workspace) -> None:
    """Write the combined forecast manifest and workbook for the last stored bar."""
    report = StageReport(6)
    summary = artifact_io.read_json(ws.node_summary_path)
    if not node_summary_is_current(ws, summary):
        report.line("summary is missing or stale; run summarize first")
        report.completed()
        return

    summary_sha256 = artifact_io.file_sha256(ws.node_summary_path)
    baseline_cube = artifact_io.load_surface(ws.baseline_cube)
    barriers = np.asarray(baseline_cube["barriers"], dtype=float)
    horizons = np.asarray(baseline_cube["horizons"], dtype=int)
    as_of = str(baseline_cube["index"][-1])
    counts = summary["summary"]
    report.line(
        f"checking {counts['bins_cleared']} cleared bins across {counts['nodes_cleared']} nodes "
        f"on the last stored bar, {as_of}"
    )

    cubes = {node["node"]: materialized_shift(ws, node["node"]) for node in summary["nodes"]}
    conditions = active_conditions(summary, cubes, baseline_cube["index"])
    used = [row for row in conditions if row["used"]]

    baseline_hits = baseline_cube["bin_hit_counts"][:, 0, :]
    baseline_counts = baseline_cube["bin_observation_counts"][0]
    combined = combination.naive_bayes_probability(
        baseline_hits, baseline_counts,
        [cubes[row["node"]]["bin_hit_counts"][:, row["bin"], :] for row in used],
        [cubes[row["node"]]["bin_observation_counts"][row["bin"]] for row in used],
    )
    baseline = combination.smoothed_probability(baseline_hits, baseline_counts[None, :])
    combined_counts = np.min(
        [baseline_counts] + [cubes[row["node"]]["bin_observation_counts"][row["bin"]] for row in used],
        axis=0,
    )

    outcomes = RunContext(ws).observed_outcomes(
        market_data_from_artifact(baseline_cube), barriers, horizons
    )
    selected = horizons - 1
    price_eligible = (np.isfinite(outcomes["downside_excursion"][selected])
                      & np.isfinite(outcomes["upside_excursion"][selected]))
    holds = np.ones(len(baseline_cube["index"]), dtype=bool)
    for row in used:
        holds &= feature_bins_from_artifact(cubes[row["node"]]) == row["bin"]
    joint, joint_counts = combination.joint_touch_rate(holds, outcomes["touch_mask"], price_eligible)
    violations = combination.nesting_violations(combined, barriers, horizons)

    forecast = {
        "workspace": ws.dir.name,
        "generated": datetime.now(timezone.utc).isoformat(),
        "artifact": "06_forecast",
        "complete": True,
        "source": ws.node_summary_path.relative_to(ws.dir).as_posix(),
        "summary_sha256": summary_sha256,
        "method": METHOD,
        "as_of": as_of,
        "summary": {
            "bins_cleared": counts["bins_cleared"], "bins_active": len(conditions),
            "conditions_used": len(used), "nesting_violations": violations,
            "expected_by_chance": counts["expected_by_chance"], "bins_tested": counts["bins_tested"],
        },
        "conditions": conditions,
        "barriers": barriers.tolist(),
        "horizons": horizons.tolist(),
        "combined_probability": _json_surface(combined),
        "combined_observation_counts": combined_counts.astype(int).tolist(),
        "baseline_probability": _json_surface(baseline),
        "joint_probability": _json_surface(joint),
        "joint_observation_counts": joint_counts.astype(int).tolist(),
    }
    if artifact_io.file_sha256(ws.node_summary_path) != summary_sha256:
        raise RuntimeError("Stage 5 summary changed during forecast; rerun forecast")
    artifact_io.write_json(ws.forecast_path, forecast)

    warnings = []
    try:
        workbooks.write_forecast_xlsx(
            ws.forecast_workbook_path, as_of=as_of, barriers=barriers, horizons=horizons,
            unit=ws.horizon_unit, combined=combined, combined_counts=combined_counts,
            baseline=baseline, joint=joint, joint_counts=joint_counts, conditions=conditions,
        )
    except PermissionError:
        warnings.append("forecast workbook locked; close it in Excel and re-run")

    report.summary(
        f"active: {len(conditions)} of {counts['bins_cleared']} cleared bins; "
        f"{len(used)} used, at most one per family"
    )
    for row in conditions:
        status = "used   " if row["used"] else "skipped"
        report.line(
            f"{status} {row['node']}  bin {row['bin_number']} of {row['bin_count']}: "
            f"{row['bin_label']}   value {row['latest_value']:.4g}   rank {row['rank']}   "
            f"p={row['monte_carlo_p_value']:.4f}   q={row['q_value']:.3f}"
        )
        if row["used"]:
            best = shift.evaluate(
                cubes[row["node"]], ws.min_dev, ws.min_bin_n, ws.min_run, bins=[row["bin"]]
            )["best"]
            report.line(f"        {strongest_cell_line(ws, best) if best else no_strong_cell_line(ws)}")
        else:
            report.line(f"        {row['note']}")

    shift_from_baseline = combined - baseline
    if np.isfinite(shift_from_baseline).any():
        i, j = np.unravel_index(np.nanargmax(np.abs(shift_from_baseline)), shift_from_baseline.shape)
        report.line(
            f"naive Bayes: largest shift {shift_from_baseline[i, j]:+.1%} at barrier "
            f"{barriers[i]:+.0%}, +{horizons[j]}{ws.horizon_unit} "
            f"({combined[i, j]:.1%} vs {baseline[i, j]:.1%} baseline)"
        )
    gap = combined - joint
    if np.isfinite(gap).any():
        report.line(
            f"historical joint: {int(joint_counts[0])} bars at +{horizons[0]}{ws.horizon_unit} "
            f"where every used condition held; naive Bayes minus joint averages "
            f"{np.nanmean(gap):+.1%}, largest gap {np.nanmax(np.abs(gap)):.1%}"
        )
    else:
        report.line(
            f"historical joint: fewer than {MIN_BIN_N} bars where every used condition held; "
            "no calibration check possible"
        )
    report.line(f"nesting: {violations} adjacent cells break barrier or horizon ordering")
    report.line(
        f"about {counts['expected_by_chance']:.1f} of {counts['bins_tested']} bins would clear by chance; "
        "shifts are in-sample estimates for bins selected as extreme"
    )
    workbook = 0 if warnings else 1
    report.line(
        f"wrote 1 forecast + {workbook} workbook → {ws.stage_dir('forecast').relative_to(ws.root_dir)}"
    )
    report.completed()
    StageReport.warnings(warnings)
