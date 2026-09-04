"""Stage 5 command: walk-forward condition composition."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from engine import bayes
from pipeline.runtime import baseline_surface, feature_panel
from workspace import Workspace

ROOT = Path(__file__).resolve().parents[1]


def cmd_bayes(ws: Workspace) -> None:
    """
    Stage 5: do the conditions compose, and does that survive out of sample?

    Everything before this measures one condition at a time and over the whole history.
    This asks the only question that follows -- what happens when several hold at once --
    and answers it under the one discipline the rest of the pipeline does not need: the
    tables, the bin edges, the prior and the scale correction are all estimated on a
    training window and applied to a later one the fit never saw.
    """
    baseline = baseline_surface(ws)
    if baseline is None:
        print("No baseline summary array - run --summary first.")
        return

    theta, horizon = ws.bayes_target(baseline)
    data, feats, fams = feature_panel(ws)

    print(
        f"\n=== 5. Composition [{ws.dir.name}] - {len(feats)} nodes, "
        f"{len(set(fams.values()))} families ==="
    )
    print(
        f"  target: P(touch {theta:+.0%} within {horizon}{ws.horizon_unit})   "
        f"expanding walk forward, {ws.bayes_folds} folds over the last half"
    )
    print(
        f"  {horizon}-bar embargo between train and test; scored on every "
        f"{horizon}th test bar so the windows do not overlap\n"
    )

    head = bayes.walk_forward(
        data, feats, fams, theta, horizon, n_bins=ws.n_bins, folds=ws.bayes_folds
    )
    m = head["metrics"]
    print(f"  {'model':<24}{'Brier':>9}{'AUC':>8}{'mean P':>9}   vs prior")
    print("  " + "-" * 60)
    ref = m["prior_only"]["brier"]
    for key, name in (
        ("prior_only", "prior only"),
        ("all_nodes", "naive Bayes, all nodes"),
        ("one_per_family", "one node per family"),
        ("one_per_family_scaled", "   + scale corrected"),
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

    th, hz = ws.summary_thetas, ws.summary_horizons
    targets = [(float(t), int(h)) for t in th if abs(t) > 1e-12 for h in hz]
    print(f"\n  sweeping the summary grid: {len(targets)} targets", end="", flush=True)
    grid = []
    for t, h in targets:
        try:
            r = bayes.walk_forward(data, feats, fams, t, h, n_bins=ws.n_bins, folds=ws.bayes_folds)
        except Exception:
            continue
        g = r["metrics"]
        grid.append({
            "theta": t,
            "horizon": h,
            "auc": g["one_per_family_scaled"]["auc"],
            "brier_prior": g["prior_only"]["brier"],
            "brier_all": g["all_nodes"]["brier"],
            "brier_dedup": g["one_per_family"]["brier"],
            "brier_scaled": g["one_per_family_scaled"]["brier"],
            "platt_a": float(np.mean([f["platt_a"] for f in r["folds"]])),
            "realized": g["realized_rate"],
            "n_scored": g["n_scored"],
        })
    beat = sum(1 for g in grid if g["brier_scaled"] < g["brier_prior"])
    print(f" - {beat} of {len(grid)} beat the prior after correction")

    np.savez_compressed(
        ws.bayes_path,
        y=head["y"],
        p_all=head["p_all"],
        p_dedup=head["p_dedup"],
        p_scaled=head["p_scaled"],
        p_prior=head["p_prior"],
        fold=head["fold"],
        grid_theta=np.array([g["theta"] for g in grid], dtype=float),
        grid_horizon=np.array([g["horizon"] for g in grid], dtype=int),
        grid_auc=np.array([g["auc"] for g in grid], dtype=float),
        grid_brier_prior=np.array([g["brier_prior"] for g in grid], dtype=float),
        grid_brier_all=np.array([g["brier_all"] for g in grid], dtype=float),
        grid_brier_dedup=np.array([g["brier_dedup"] for g in grid], dtype=float),
        grid_brier_scaled=np.array([g["brier_scaled"] for g in grid], dtype=float),
        grid_platt_a=np.array([g["platt_a"] for g in grid], dtype=float),
        grid_realized=np.array([g["realized"] for g in grid], dtype=float),
        grid_n_scored=np.array([g["n_scored"] for g in grid], dtype=int),
        meta=np.array(json.dumps({
            "workspace": ws.dir.name,
            "theta": theta,
            "horizon": horizon,
            "unit": ws.horizon_unit,
            "generated": datetime.now(timezone.utc).isoformat(),
        })),
    )
    ws.write_json(ws.bayes_summary_path, {
        "workspace": ws.dir.name,
        "generated": datetime.now(timezone.utc).isoformat(),
        "target": {"theta": theta, "horizon": horizon, "unit": ws.horizon_unit},
        "design": {
            "folds": ws.bayes_folds,
            "embargo_bars": horizon,
            "scored_every": horizon,
            "n_features": head["n_features"],
            "n_families": len(set(fams.values())),
            "shrinkage_k": bayes.SHRINK_K,
            "note": "bin edges, per-bin rates, the prior and the scale correction "
                    "are all fit on the training window only",
        },
        "metrics": m,
        "platt_a_mean": a_mean,
        "folds": head["folds"],
        "kept_per_fold": head["kept"],
        "grid": grid,
        "grid_beat_prior": beat,
    })
    print(f"\n  wrote {ws.bayes_path.relative_to(ROOT)} and {ws.bayes_summary_path.relative_to(ROOT)}")
