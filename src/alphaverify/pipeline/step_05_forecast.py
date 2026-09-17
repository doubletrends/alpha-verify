"""Stage 5: which nodes cleared, and the forecast from those active on the last stored bar."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone

import numpy as np

from alphaverify.domain import barrier, combination, shift
from alphaverify.domain.notation import MIN_BIN_N
from alphaverify.infrastructure import artifact_io
from alphaverify.infrastructure.artifact_history import (
    feature_bins_from_artifact, market_data_from_artifact,
)
from alphaverify.infrastructure.workspace import STAGE_DIRECTORIES, Workspace
from alphaverify.pipeline.context import RunContext, materialized_shift
from alphaverify.pipeline.reporting import StageReport
from alphaverify.pipeline.step_04_selection import selection_summary_is_current
from alphaverify.presentation import workbooks
from alphaverify.presentation.display import format_barrier

_BIN_FIELDS = ("bin", "bin_number", "bin_label", "rank", "monte_carlo_p_value", "q_value")

METHOD = {
    "active": "the last stored bar's feature value falls in a cleared bin under stored Stage 1 edges",
    "family_rule": "at most one active bin per family: best rank, then node and bin",
    "combination": "naive Bayes per cell: logit baseline + sum(logit condition - logit baseline); "
                   "conditions treated as independent given the outcome",
    "smoothing": "(hits + 1) / (n + 2) before log-odds",
    "min_bin_n": MIN_BIN_N,
    "joint": "historical touch rate on bars where every contributing condition held",
}


def cleared_nodes(selected: list[dict], tested_bins: Counter) -> list[dict]:
    """Group ranked selected bins by node; nodes follow their strongest bin's rank."""
    nodes: dict[str, dict] = {}
    for row in selected:
        node = nodes.setdefault(row["node"], {
            "node": row["node"], "family": row["family"],
            "feature": row["feature"], "params": row["params"],
            "bins_tested": tested_bins[row["node"]], "bins_cleared": 0, "bins": [],
        })
        node["bins_cleared"] += 1
        node["bins"].append({field: row[field] for field in _BIN_FIELDS})
    for node in nodes.values():
        node["bins"].sort(key=lambda item: item["bin"])
        node["best_rank"] = min(item["rank"] for item in node["bins"])
        node["best_p_value"] = min(item["monte_carlo_p_value"] for item in node["bins"])
        node["best_q_value"] = min(item["q_value"] for item in node["bins"])
    return sorted(nodes.values(), key=lambda node: (node["best_rank"], node["node"]))


def active_conditions(nodes: list[dict], cubes: dict[str, dict], baseline_index) -> list[dict]:
    """Cleared bins holding on the last stored bar, marked used under the one-per-family rule."""
    active = []
    for node in nodes:
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


def _strongest_cell_line(ws: Workspace, best: dict | None) -> str:
    """One used condition's strongest cell from ``shift.evaluate``, or why it has none."""
    if not best:
        return f"no cell reaches {ws.min_dev:.1%} across {ws.min_run} adjacent barrier rows"
    return (
        f"strongest cell: shift={best['dev']:+.1%}; "
        f"P={best['conditional_probability']:.1%} "
        f"vs {best['baseline_probability']:.1%} baseline "
        f"at barrier={format_barrier(best['barrier'], ws.barriers)}, +{best['horizon']}{ws.horizon_unit} "
        f"(run={best['run']}, n={best['bin_observation_count']})"
    )


def _json_surface(values: np.ndarray) -> list:
    return [[None if not np.isfinite(value) else float(value) for value in row] for row in values]


def cmd_forecast(ws: Workspace) -> None:
    """Write the cleared nodes and the combined forecast for the last stored bar."""
    report = StageReport("forecast")
    selection = artifact_io.read_json(ws.selection_summary_path)
    if not selection_summary_is_current(ws, selection):
        report.line("selection is missing or stale; run select first")
        report.completed()
        return

    selection_sha256 = artifact_io.file_sha256(ws.selection_summary_path)
    validation = artifact_io.read_json(ws.validation_summary_path)
    tested_bins = Counter(row["node"] for row in validation.get("tests", []))
    nodes = cleared_nodes(selection.get("selected", []), tested_bins)
    bins_tested = sum(tested_bins.values())
    bins_cleared = sum(node["bins_cleared"] for node in nodes)
    expected_by_chance = selection["summary"]["expected_by_chance"]

    baseline_cube = artifact_io.load_surface(ws.baseline_cube)
    barriers = np.asarray(baseline_cube["barriers"], dtype=float)
    horizons = np.asarray(baseline_cube["horizons"], dtype=int)
    as_of = str(baseline_cube["index"][-1])
    report.line(
        f"checking {bins_cleared} cleared bins across {len(nodes)} nodes "
        f"on the last stored bar, {as_of}"
    )

    cubes = {node["node"]: materialized_shift(ws, node["node"]) for node in nodes}
    conditions = active_conditions(nodes, cubes, baseline_cube["index"])
    used = [row for row in conditions if row["used"]]

    baseline_hits = baseline_cube["bin_hit_counts"][:, 0, :]
    baseline_counts = baseline_cube["bin_observation_counts"][0]
    combined, combined_counts = combination.naive_bayes_probability(
        baseline_hits, baseline_counts,
        [cubes[row["node"]]["bin_hit_counts"][:, row["bin"], :] for row in used],
        [cubes[row["node"]]["bin_observation_counts"][row["bin"]] for row in used],
    )
    baseline = combination.smoothed_probability(baseline_hits, baseline_counts[None, :])

    outcomes = RunContext(ws).observed_outcomes(
        market_data_from_artifact(baseline_cube), barriers, horizons
    )
    holds = np.ones(len(baseline_cube["index"]), dtype=bool)
    for row in used:
        holds &= feature_bins_from_artifact(cubes[row["node"]]) == row["bin"]
    joint, joint_counts = combination.joint_touch_rate(
        holds, outcomes["touch_mask"], barrier.observed_price_eligibility(outcomes, horizons)
    )
    violations = combination.nesting_violations(combined, barriers, horizons)

    forecast = {
        "workspace": ws.dir.name,
        "generated": datetime.now(timezone.utc).isoformat(),
        "artifact": STAGE_DIRECTORIES["forecast"],
        "complete": True,
        "source": ws.selection_summary_path.relative_to(ws.dir).as_posix(),
        "selection_sha256": selection_sha256,
        "method": METHOD,
        "as_of": as_of,
        "summary": {
            "nodes_tested": len(tested_bins), "nodes_cleared": len(nodes),
            "bins_tested": bins_tested, "bins_cleared": bins_cleared,
            "expected_by_chance": expected_by_chance,
            "bins_active": len(conditions), "conditions_used": len(used),
            "nesting_violations": violations,
        },
        "nodes": nodes,
        "conditions": conditions,
        "barriers": barriers.tolist(),
        "horizons": horizons.tolist(),
        "combined_probability": _json_surface(combined),
        "combined_observation_counts": combined_counts.astype(int).tolist(),
        "baseline_probability": _json_surface(baseline),
        "joint_probability": _json_surface(joint),
        "joint_observation_counts": joint_counts.astype(int).tolist(),
    }
    if artifact_io.file_sha256(ws.selection_summary_path) != selection_sha256:
        raise RuntimeError("Stage 4 selection changed during forecast; rerun forecast")
    artifact_io.write_json(ws.forecast_path, forecast)

    warnings = []
    try:
        workbooks.write_forecast_xlsx(
            ws.forecast_workbook_path, as_of=as_of, barriers=barriers, horizons=horizons,
            unit=ws.horizon_unit, combined=combined, combined_counts=combined_counts,
            baseline=baseline, joint=joint, joint_counts=joint_counts,
            nodes=nodes, conditions=conditions,
        )
    except PermissionError:
        warnings.append("forecast workbook locked; close it in Excel and re-run")

    report.summary(
        f"{len(nodes)} of {len(tested_bins)} tested nodes cleared {bins_cleared} bins; "
        f"{len(conditions)} active on the last bar, {len(used)} used, at most one per family"
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
            report.line(f"        {_strongest_cell_line(ws, best)}")
        else:
            report.line(f"        {row['note']}")

    shift_from_baseline = combined - baseline
    if np.isfinite(shift_from_baseline).any():
        i, j = np.unravel_index(np.nanargmax(np.abs(shift_from_baseline)), shift_from_baseline.shape)
        report.line(
            f"naive Bayes: largest shift {shift_from_baseline[i, j]:+.1%} at barrier "
            f"{format_barrier(barriers[i], barriers)}, +{horizons[j]}{ws.horizon_unit} "
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
        f"about {expected_by_chance:.1f} of {bins_tested} bins would clear by chance; "
        "shifts are in-sample estimates for bins selected as extreme"
    )
    workbook = 0 if warnings else 1
    report.line(
        f"wrote 1 forecast + {workbook} workbook → {ws.stage_dir('forecast').relative_to(ws.root_dir)}"
    )
    report.completed()
    StageReport.warnings(warnings)
