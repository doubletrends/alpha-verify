"""Stage 6 command: walk-forward condition composition."""

from __future__ import annotations

import json
from datetime import datetime, timezone
import numpy as np

from domain import bayes, shift
from infrastructure.workspace import BASELINE_NODE, Workspace
from pipeline.context import artifact_feature_panel
from pipeline.step_04_validation import validation_summary_is_current
from presentation import workbooks


def composition_targets(ws: Workspace) -> list[tuple[float, int]]:
    """Every non-zero barrier and horizon pair on the workspace grid."""
    return [
        (float(delta), int(horizon))
        for delta in ws.deltas
        if abs(delta) > 1e-12
        for horizon in ws.horizons
    ]


def cmd_bayes(ws: Workspace) -> None:
    """
    Stage 6: do the best training-window node tables compose out of sample?

    Everything before this measures one condition at a time and over the whole history.
    This asks the only question that follows -- what happens when several hold at once --
    and answers it under the one discipline the rest of the pipeline does not need: the
    tables, the bin edges, the prior and the scale correction are all estimated on a
    training window and applied to a later one the fit never saw.
    """
    baseline_path = ws.shift_cube_path("_base", BASELINE_NODE)
    if not baseline_path.exists():
        print("No baseline shift array - run shift first.")
        return

    baseline = shift.load(baseline_path)["base"]
    Δ, horizon = ws.bayes_target(baseline)
    validation = ws.read_json(ws.validation_summary_path)
    if not validation_summary_is_current(ws, validation):
        print("No current complete validation summary - run validation first.")
        return
    try:
        data, feats, fams = artifact_feature_panel(ws)
    except ValueError as e:
        print(f"Cannot run composition from artifacts: {e}")
        return
    cleared_ids = {row["node"] for row in validation.get("cleared", [])}
    forecast_feats = {name: feature for name, feature in feats.items() if name in cleared_ids}
    if not forecast_feats:
        print("No validation-cleared nodes are available for the current Bayes surface.")
        return

    print(
        f"\n=== 6. Composition [{ws.dir.name}] - {len(feats)} candidate nodes, "
        f"{len(set(fams.values()))} families ==="
    )
    print(
        f"  target: P(touch {Δ:+.0%} within {horizon}{ws.horizon_unit})   "
        f"expanding walk forward, {ws.bayes_folds} folds over the last half"
    )
    print(
        f"  {horizon}-bar embargo between train and test; scored on every "
        f"{horizon}th test bar so the windows do not overlap\n"
    )
    print(
        f"  each fold ranks whole ten-bin tables on its training history and keeps "
        f"the top {ws.selection_top_k} distinct nodes\n"
    )

    head = bayes.walk_forward(
        data, feats, fams, Δ, horizon, n_bins=ws.n_bins, folds=ws.bayes_folds,
        top_k=ws.selection_top_k, weighted=True,
    )
    m = head["metrics"]
    print(f"  {'model':<24}{'Brier':>9}{'AUC':>8}{'mean P':>9}   vs prior")
    print("  " + "-" * 60)
    ref = m["prior_only"]["brier"]
    for key, name in (
        ("prior_only", "prior only"),
        ("all_nodes", "naive Bayes, top nodes"),
        ("one_per_family", "one node per family"),
        ("one_per_family_scaled", "   + scale corrected"),
        ("weighted_nodes", "weighted redundancy-aware"),
    ):
        e = m[key]
        mark = "" if key == "prior_only" else f"{(e['brier'] / ref - 1) * 100:+.1f}%"
        auc_s = "     -  " if not np.isfinite(e["auc"]) else f"{e['auc']:>8.3f}"
        print(f"  {name:<24}{e['brier']:>9.4f}{auc_s}{e['mean_predicted']:>9.1%}   {mark}")
    a_mean = float(np.mean([f["platt_a"] for f in head["folds"]]))
    print(
        f"\n  realized {m['realized_rate']:.1%} over {m['n_scored']} non-overlapping "
        "out-of-sample bars"
    )
    print(
        f"  scale correction a = {a_mean:.3f}: the raw score is overconfident by "
        f"{1 / a_mean:.1f}x"
    )

    targets = composition_targets(ws)
    surface_deltas = np.asarray(ws.deltas, dtype=float)
    surface_horizons = np.asarray(ws.horizons, dtype=int)
    delta_indexes = {float(value): index for index, value in enumerate(surface_deltas)}
    horizon_indexes = {int(value): index for index, value in enumerate(surface_horizons)}

    print(
        f"\n  fitting {len(surface_deltas) * len(surface_horizons)} current forecasts",
        end=" ... ",
        flush=True,
    )
    current_surface = bayes.current_weighted_surface(
        data,
        forecast_feats,
        surface_deltas,
        surface_horizons,
        n_bins=ws.n_bins,
        top_k=ws.selection_top_k,
    )
    current_probability = current_surface["probability"]
    current_probability_raw = current_surface["raw_probability"]
    current_n = current_surface["n_observations"]
    current_as_of = current_surface["as_of"]
    if baseline.shape != current_probability.shape:
        raise ValueError(
            "Stage 2 baseline shape does not match the Bayes surface: "
            f"{baseline.shape} vs {current_probability.shape}"
        )
    current_shift = (current_probability - baseline) * 100.0
    current_shift_raw = (current_probability_raw - baseline) * 100.0
    print("done")

    demonstration_surface = None
    if ws.demonstration_date is not None:
        print(
            f"  fitting historical Bayes demonstration as of {ws.demonstration_date}",
            end=" ... ",
            flush=True,
        )
        demonstration_surface = bayes.current_weighted_surface(
            data,
            forecast_feats,
            surface_deltas,
            surface_horizons,
            n_bins=ws.n_bins,
            top_k=ws.selection_top_k,
            as_of=ws.demonstration_date,
            ridge_fallback="prior",
        )
        print("done")

    print(
        f"\n  sweeping the full shift grid: {len(targets)} walk-forward validation targets",
        end="",
        flush=True,
    )
    grid = []
    for delta, sweep_horizon in targets:
        delta_index = delta_indexes[delta]
        horizon_index = horizon_indexes[sweep_horizon]
        try:
            r = bayes.walk_forward(
                data,
                feats,
                fams,
                delta,
                sweep_horizon,
                n_bins=ws.n_bins,
                folds=ws.bayes_folds,
                top_k=ws.selection_top_k,
            )
        except Exception:
            continue
        g = r["metrics"]
        grid.append({
            "Δ": delta,
            "horizon": sweep_horizon,
            "auc": g["one_per_family_scaled"]["auc"],
            "brier_prior": g["prior_only"]["brier"],
            "brier_all": g["all_nodes"]["brier"],
            "brier_dedup": g["one_per_family"]["brier"],
            "brier_scaled": g["one_per_family_scaled"]["brier"],
            "platt_a": float(np.mean([f["platt_a"] for f in r["folds"]])),
            "realized": g["realized_rate"],
            "n_scored": g["n_scored"],
            "current_probability": float(current_probability_raw[delta_index, horizon_index]),
        })
    beat = sum(1 for g in grid if g["brier_scaled"] < g["brier_prior"])
    print(f" - {beat} of {len(grid)} beat the prior after correction")

    try:
        workbooks.write_bayes_xlsx(
            ws.bayes_workbook_path,
            current_probability,
            surface_deltas,
            surface_horizons,
            current_n,
            ws.horizon_unit,
        )
        workbooks.write_bayes_shift_xlsx(
            ws.bayes_shift_workbook_path,
            current_shift,
            current_probability,
            baseline,
            surface_deltas,
            surface_horizons,
            current_n,
            ws.horizon_unit,
        )
    except PermissionError:
        print("A Bayes workbook is locked; close it in Excel and run bayes again.")
        return

    ws.bayes_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        ws.bayes_path,
        y=head["y"],
        p_all=head["p_all"],
        p_dedup=head["p_dedup"],
        p_scaled=head["p_scaled"],
        p_weighted=head["p_weighted"],
        p_prior=head["p_prior"],
        fold=head["fold"],
        grid_Δ=np.array([g["Δ"] for g in grid], dtype=float),
        grid_horizon=np.array([g["horizon"] for g in grid], dtype=int),
        grid_auc=np.array([g["auc"] for g in grid], dtype=float),
        grid_brier_prior=np.array([g["brier_prior"] for g in grid], dtype=float),
        grid_brier_all=np.array([g["brier_all"] for g in grid], dtype=float),
        grid_brier_dedup=np.array([g["brier_dedup"] for g in grid], dtype=float),
        grid_brier_scaled=np.array([g["brier_scaled"] for g in grid], dtype=float),
        grid_platt_a=np.array([g["platt_a"] for g in grid], dtype=float),
        grid_realized=np.array([g["realized"] for g in grid], dtype=float),
        grid_n_scored=np.array([g["n_scored"] for g in grid], dtype=int),
        current_probability=current_probability,
        current_probability_raw=current_probability_raw,
        current_shift=current_shift,
        current_shift_raw=current_shift_raw,
        current_baseline=baseline,
        current_n=current_n,
        current_Δ=surface_deltas,
        current_horizon=surface_horizons,
        current_as_of=np.array(current_as_of),
        current_node_ids=np.asarray(list(forecast_feats), dtype=str),
        demonstration_probability=(
            demonstration_surface["probability"]
            if demonstration_surface is not None else np.empty((0, 0), dtype=float)
        ),
        demonstration_probability_raw=(
            demonstration_surface["raw_probability"]
            if demonstration_surface is not None else np.empty((0, 0), dtype=float)
        ),
        demonstration_n=(
            demonstration_surface["n_observations"]
            if demonstration_surface is not None else np.empty(0, dtype=int)
        ),
        demonstration_as_of=np.array(
            demonstration_surface["as_of"] if demonstration_surface is not None else ""
        ),
        demonstration_ridge_fallback_cells=np.array(
            demonstration_surface["ridge_fallback_cells"]
            if demonstration_surface is not None else 0,
            dtype=int,
        ),
        meta=np.array(json.dumps({
            "workspace": ws.dir.name,
            "Δ": Δ,
            "horizon": horizon,
            "unit": ws.horizon_unit,
            "generated": datetime.now(timezone.utc).isoformat(),
        })),
    )
    ws.write_json(ws.bayes_summary_path, {
        "workspace": ws.dir.name,
        "generated": datetime.now(timezone.utc).isoformat(),
        "target": {"Δ": Δ, "horizon": horizon, "unit": ws.horizon_unit},
        "design": {
            "folds": ws.bayes_folds,
            "embargo_bars": horizon,
            "scored_every": horizon,
            "n_candidates": head["n_candidates"],
            "top_k": ws.selection_top_k,
            "n_families": len(set(fams.values())),
            "shrinkage_k": bayes.SHRINK_K,
            "weighted_model": "non-negative ridge logistic weights over cross-fitted node contributions",
            "note": "bin edges, per-bin rates, node selection, the prior and the "
                    "scale correction are all fit on the training window only",
        },
        "metrics": m,
        "platt_a_mean": a_mean,
        "folds": head["folds"],
        "selected_per_fold": head["selected"],
        "kept_per_fold": head["kept"],
        "weighted_folds": [
            {
                "fold": fold["fold"],
                "ridge": fold.get("weighted_ridge"),
                "calibration_bars": fold.get("weighted_calibration_bars"),
                "nodes": fold.get("weighted_nodes", {}),
                "redundancy_clusters": fold.get("redundancy_clusters", []),
            }
            for fold in head["folds"]
        ],
        "grid": grid,
        "grid_beat_prior": beat,
        "current_surface": {
            "as_of": current_as_of,
            "model": "target-specific non-negative ridge weights over cross-fitted node contributions",
            "coherence": "alternating isotonic projection over nested barriers and horizons",
            "source": "Stage 4 validation-cleared nodes",
            "nodes": list(forecast_feats),
            "n_nodes": len(forecast_feats),
            "rows": len(surface_deltas),
            "columns": len(surface_horizons),
            "minimum_probability": float(np.min(current_probability)),
            "maximum_probability": float(np.max(current_probability)),
            "ridge_fallback_cells": current_surface["ridge_fallback_cells"],
            "ridge_fallback": current_surface["ridge_fallback"],
            "workbook": str(ws.bayes_workbook_path.relative_to(ws.dir)).replace("\\", "/"),
            "shift_workbook": str(
                ws.bayes_shift_workbook_path.relative_to(ws.dir)
            ).replace("\\", "/"),
            "shift_definition": "100 * (weighted Bayes probability - unconditional baseline probability)",
            "minimum_shift_pp": float(np.min(current_shift)),
            "maximum_shift_pp": float(np.max(current_shift)),
        },
        "demonstration_surface": (
            None if demonstration_surface is None else {
                "requested_as_of": ws.demonstration_date,
                "as_of": demonstration_surface["as_of"],
                "model": "target-specific non-negative ridge weights over cross-fitted node contributions",
                "coherence": "alternating isotonic projection over nested barriers and horizons",
                "source": "Stage 4 validation-cleared nodes",
                "nodes": list(forecast_feats),
                "n_nodes": len(forecast_feats),
                "rows": len(surface_deltas),
                "columns": len(surface_horizons),
                "ridge_fallback_cells": demonstration_surface["ridge_fallback_cells"],
                "ridge_fallback": "target historical prior",
                "minimum_probability": float(np.min(demonstration_surface["probability"])),
                "maximum_probability": float(np.max(demonstration_surface["probability"])),
            }
        ),
    })
    print(f"\n  wrote {ws.bayes_path.relative_to(ws.root_dir)}")
    print(f"  wrote {ws.bayes_summary_path.relative_to(ws.root_dir)}")
    print(f"  wrote {ws.bayes_workbook_path.relative_to(ws.root_dir)}")
    print(f"  wrote {ws.bayes_shift_workbook_path.relative_to(ws.root_dir)}")
