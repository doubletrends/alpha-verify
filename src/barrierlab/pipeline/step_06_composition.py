"""Stage 6 command: walk-forward condition composition."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import numpy as np
import pandas as pd

from barrierlab.domain import bayes
from barrierlab.infrastructure import artifact_io
from barrierlab.infrastructure.workspace import Workspace
from barrierlab.presentation import workbooks


def composition_targets(ws: Workspace) -> list[tuple[float, int]]:
    """Every non-zero barrier and horizon pair on the workspace grid."""
    return [
        (float(delta), int(horizon))
        for delta in ws.deltas
        if abs(delta) > 1e-12
        for horizon in ws.horizons
    ]


def composition_feature_panel(ws: Workspace) -> tuple[pd.DataFrame, dict, dict, dict]:
    """Rebuild composition inputs exclusively from Stage 5's handoff artifact."""
    if not ws.composition_inputs_path.exists():
        raise ValueError("missing Stage 5 composition inputs - run redundancy first")
    bundle = artifact_io.load_composition_inputs(ws.composition_inputs_path)
    required = {"index", "high", "low", "close", "base", "Δs", "horizons", "node_ids", "families"}
    if required - set(bundle):
        raise ValueError("Stage 5 composition inputs are incomplete - run redundancy first")
    data = pd.DataFrame({
        "high": bundle["high"].astype(float),
        "low": bundle["low"].astype(float),
        "close": bundle["close"].astype(float),
    }, index=pd.to_datetime(bundle["index"]))
    names = [str(value) for value in bundle["node_ids"]]
    feats = {
        name: pd.Series(bundle[f"feature_{index}"].astype(float), index=data.index)
        for index, name in enumerate(names)
        if f"feature_{index}" in bundle
    }
    fams = {name: str(bundle["families"][index]) for index, name in enumerate(names)}
    if not feats:
        raise ValueError("Stage 5 composition inputs contain no feature histories")
    return data, feats, fams, bundle


def cmd_composition(ws: Workspace) -> None:
    """
    Stage 6: do the best training-window node tables compose out of sample?

    Everything before this measures one condition at a time and over the whole history.
    This asks the only question that follows -- what happens when several hold at once --
    and answers it under the one discipline the rest of the pipeline does not need: the
    tables, the bin edges, the prior and the scale correction are all estimated on a
    training window and applied to a later one the fit never saw.
    """
    try:
        data, feats, fams, inputs = composition_feature_panel(ws)
    except ValueError as e:
        print(f"Cannot run composition from Stage 5: {e}")
        return
    report_arrays = {
        f"report_{key}": value
        for key, value in inputs.items()
        if key in {"index", "high", "low", "close", "node_ids", "families", "validation_json"}
        or key.startswith(("feature_", "shift_", "edges_"))
    }
    baseline = inputs["base"]
    Δ, horizon = ws.composition_target(baseline)
    representative_ids = [str(value) for value in inputs["representative_node_ids"]]
    forecast_feats = {name: feats[name] for name in representative_ids if name in feats}
    fold_representatives = json.loads(str(np.asarray(inputs["fold_representatives_json"]).item()))
    if not forecast_feats or len(fold_representatives) != ws.composition_folds:
        print("Stage 5 representative plan is incomplete - run redundancy first.")
        return

    print(
        f"\n=== 6. Composition [{ws.dir.name}] - {len(forecast_feats)} cluster representatives ==="
    )
    print(
        f"  target: P(touch {Δ:+.0%} within {horizon}{ws.horizon_unit})   "
        f"expanding walk forward, {ws.composition_folds} folds over the last half"
    )
    print(
        f"  {horizon}-bar embargo between train and test; scored on every "
        f"{horizon}th test bar so the windows do not overlap\n"
    )
    print(
        "  Stage 5 selected one representative per redundancy cluster in each training fold; "
        "all retained nodes have equal Naive-Bayes weight\n"
    )

    head = bayes.walk_forward(
        data, feats, fams, Δ, horizon, n_bins=ws.n_bins, folds=ws.composition_folds,
        top_k=None, kept_names_by_fold=fold_representatives,
    )
    m = head["metrics"]
    print(f"  {'model':<24}{'Brier':>9}{'AUC':>8}{'mean P':>9}   vs prior")
    print("  " + "-" * 60)
    ref = m["prior_only"]["brier"]
    for key, name in (("prior_only", "prior only"), ("one_per_family", "equal-weight cluster representatives")):
        e = m[key]
        mark = "" if key == "prior_only" else f"{(e['brier'] / ref - 1) * 100:+.1f}%"
        auc_s = "     -  " if not np.isfinite(e["auc"]) else f"{e['auc']:>8.3f}"
        print(f"  {name:<24}{e['brier']:>9.4f}{auc_s}{e['mean_predicted']:>9.1%}   {mark}")
    print(
        f"\n  realized {m['realized_rate']:.1%} over {m['n_scored']} non-overlapping "
        "out-of-sample bars"
    )
    surface_deltas = np.asarray(ws.deltas, dtype=float)
    surface_horizons = np.asarray(ws.horizons, dtype=int)

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
        top_k=None,
        equal_weight=True,
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

    report_as_of = ws.report_as_of
    print(
        f"  fitting report Bayes surface as of {report_as_of}",
        end=" ... ",
        flush=True,
    )
    demonstration_surface = bayes.current_weighted_surface(
        data,
        forecast_feats,
        surface_deltas,
        surface_horizons,
        n_bins=ws.n_bins,
        top_k=None,
        as_of=report_as_of,
        equal_weight=True,
    )
    print("done")

    grid = []

    try:
        workbooks.write_bayes_xlsx(
            ws.composition_probability_workbook_path,
            current_probability,
            surface_deltas,
            surface_horizons,
            current_n,
            ws.horizon_unit,
        )
        workbooks.write_bayes_shift_xlsx(
            ws.composition_shift_workbook_path,
            current_shift,
            current_probability,
            baseline,
            surface_deltas,
            surface_horizons,
            current_n,
            ws.horizon_unit,
        )
    except PermissionError:
        print(
            "A composition workbook is locked; close it in Excel and run "
            "composition again."
        )
        return

    artifact_io.save_composition(
        ws.composition_array_path,
        y=head["y"],
        p_all=head["p_all"],
        p_dedup=head["p_dedup"],
        p_scaled=head["p_scaled"],
        p_prior=head["p_prior"],
        fold=head["fold"],
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
            0,
            dtype=int,
        ),
        **report_arrays,
        meta={
            "workspace": ws.dir.name,
            "Δ": Δ,
            "horizon": horizon,
            "unit": ws.horizon_unit,
            "generated": datetime.now(timezone.utc).isoformat(),
        },
    )
    ws.write_json(ws.composition_summary_path, {
        "workspace": ws.dir.name,
        "generated": datetime.now(timezone.utc).isoformat(),
        "target": {"Δ": Δ, "horizon": horizon, "unit": ws.horizon_unit},
        "design": {
            "folds": ws.composition_folds,
            "embargo_bars": horizon,
            "scored_every": horizon,
            "n_candidates": head["n_candidates"],
            "representative_selection": "one conditional-NMI cluster representative per fold",
            "n_families": len(set(fams.values())),
            "shrinkage_k": bayes.SHRINK_K,
            "model": "equal-weight Naive Bayes over Stage 5 cluster representatives",
            "note": "bin edges, rates, and fold-local representatives are fit before each test block",
        },
        "metrics": m,
        "folds": head["folds"],
        "selected_per_fold": head["selected"],
        "kept_per_fold": head["kept"],
        "report_validation": json.loads(str(np.asarray(inputs["validation_json"]).item())),
        "current_surface": {
            "as_of": current_as_of,
            "model": "equal-weight Naive Bayes over Stage 5 cluster representatives",
            "coherence": "alternating isotonic projection over nested barriers and horizons",
            "source": "Stage 4 validation-cleared nodes",
            "nodes": list(forecast_feats),
            "n_nodes": len(forecast_feats),
            "rows": len(surface_deltas),
            "columns": len(surface_horizons),
            "minimum_probability": float(np.min(current_probability)),
            "maximum_probability": float(np.max(current_probability)),
            "workbook": str(
                ws.composition_probability_workbook_path.relative_to(ws.dir)
            ).replace("\\", "/"),
            "shift_workbook": str(
                ws.composition_shift_workbook_path.relative_to(ws.dir)
            ).replace("\\", "/"),
            "shift_definition": "100 * (equal-weight Bayes probability - unconditional baseline probability)",
            "minimum_shift_pp": float(np.min(current_shift)),
            "maximum_shift_pp": float(np.max(current_shift)),
        },
        "demonstration_surface": (
            None if demonstration_surface is None else {
                "requested_as_of": report_as_of,
                "as_of": demonstration_surface["as_of"],
                "model": "equal-weight Naive Bayes over Stage 5 cluster representatives",
                "coherence": "alternating isotonic projection over nested barriers and horizons",
                "source": "Stage 4 validation-cleared nodes",
                "nodes": list(forecast_feats),
                "n_nodes": len(forecast_feats),
                "rows": len(surface_deltas),
                "columns": len(surface_horizons),
                "minimum_probability": float(np.min(demonstration_surface["probability"])),
                "maximum_probability": float(np.max(demonstration_surface["probability"])),
            }
        ),
    })
    print(f"\n  wrote {ws.composition_array_path.relative_to(ws.root_dir)}")
    print(f"  wrote {ws.composition_summary_path.relative_to(ws.root_dir)}")
    print(
        f"  wrote {ws.composition_probability_workbook_path.relative_to(ws.root_dir)}"
    )
    print(f"  wrote {ws.composition_shift_workbook_path.relative_to(ws.root_dir)}")
