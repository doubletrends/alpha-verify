"""Build a leakage-safe Stage-6-like surface at a historical volatility stress date.

This is an exploratory probe, not a numbered pipeline stage. It selects an anchor using
only a trailing volatility feature, refits every delta/horizon forecast using outcomes
completed by that date, renders the familiar probability workbook, and reruns the
bracket expectancy exhaust.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from domain import bayes  # noqa: E402
from infrastructure.workspaces.workspace import Workspace  # noqa: E402
from pipeline.context import artifact_feature_panel  # noqa: E402
from presentation import workbooks  # noqa: E402
from scripts.probe_expectancy import _print, _write_csv, exhaust_surface  # noqa: E402


def select_anchor(
    data: pd.DataFrame,
    features: dict,
    volatility_node: str,
    percentile: float,
    min_history: int,
    requested: str | None = None,
) -> tuple[int, float, float]:
    """Return an eligible anchor index, its volatility value, and stress threshold."""
    if volatility_node not in features:
        raise ValueError(f"feature panel has no {volatility_node!r}")
    if not 0.0 < percentile < 100.0:
        raise ValueError("percentile must be between 0 and 100")
    if min_history < 200:
        raise ValueError("min-history must be at least 200 bars")

    panel = np.vstack([feature.to_numpy(float) for feature in features.values()])
    complete = np.isfinite(panel).all(axis=0)
    eligible = complete & (np.arange(len(data)) >= min_history)
    volatility = features[volatility_node].to_numpy(float)
    eligible &= np.isfinite(volatility)
    candidates = np.flatnonzero(eligible)
    if not len(candidates):
        raise ValueError("no complete feature row remains after the history requirement")
    threshold = float(np.percentile(volatility[candidates], percentile))

    if requested is not None:
        before = candidates[data.index[candidates] <= pd.Timestamp(requested)]
        if not len(before):
            raise ValueError("requested anchor precedes the first eligible feature row")
        index = int(before[-1])
        return index, float(volatility[index]), threshold

    above = eligible & (volatility >= threshold)
    entries = above & ~np.r_[False, above[:-1]]
    entry_candidates = np.flatnonzero(entries)
    if not len(entry_candidates):
        entry_candidates = np.flatnonzero(above)
    index = int(entry_candidates[0])
    return index, float(volatility[index]), threshold


def build_historical_surface(
    workspace: Workspace,
    data: pd.DataFrame,
    features: dict,
    anchor_date: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, int]:
    """Refit the complete Stage 6 probability face as it was knowable at the anchor."""
    deltas = np.asarray(workspace.deltas, dtype=float)
    horizons = np.asarray(workspace.horizons, dtype=int)
    raw = np.full((len(deltas), len(horizons)), np.nan, dtype=float)
    observations = np.zeros(len(horizons), dtype=int)
    actual_as_of = None
    prior_fallback_cells = 0

    total = len(deltas) * len(horizons)
    completed = 0
    print(
        f"  fitting {total:,} historical forecasts from {len(features)} candidate nodes",
        flush=True,
    )
    for delta_index, delta in enumerate(deltas):
        for horizon_index, horizon in enumerate(horizons):
            forecast = bayes.current_weighted_forecast(
                data,
                features,
                float(delta),
                int(horizon),
                n_bins=workspace.n_bins,
                top_k=workspace.selection_top_k,
                as_of=anchor_date,
            )
            if forecast["ridge"] is None:
                # A sparse cross-fit cannot estimate reliability weights honestly.
                # Use the target's historical base rate instead of silently presenting
                # unweighted Naive Bayes as a weighted forecast.
                raw[delta_index, horizon_index] = forecast["prior"]
                prior_fallback_cells += 1
            else:
                raw[delta_index, horizon_index] = forecast["probability"]
            observations[horizon_index] = forecast["n_observations"]
            actual_as_of = forecast["as_of"]
            completed += 1
        if (delta_index + 1) % 5 == 0 or completed == total:
            print(f"    {completed:>4}/{total}", flush=True)

    if not np.isfinite(raw).all() or actual_as_of is None:
        raise ValueError("historical probability surface is incomplete")
    probability = bayes.coherent_probability_surface(raw, deltas)
    return (
        probability,
        raw,
        deltas,
        horizons,
        observations,
        actual_as_of,
        prior_fallback_cells,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe bracket expectancy at an objectively selected high-volatility date."
    )
    parser.add_argument("--workspace", default="nasdaq_daily")
    parser.add_argument("--volatility-node", default="realized_vol_14")
    parser.add_argument("--percentile", type=float, default=99.0)
    parser.add_argument("--min-history", type=int, default=2500)
    parser.add_argument("--anchor", help="optional YYYY-MM-DD override")
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    workspace = Workspace(args.workspace)
    data, features, _ = artifact_feature_panel(workspace)
    anchor_index, volatility, threshold = select_anchor(
        data,
        features,
        args.volatility_node,
        args.percentile,
        args.min_history,
        args.anchor,
    )
    anchor_date = str(data.index[anchor_index])[:10]
    mode = "manual" if args.anchor else f"first entry into top {100.0 - args.percentile:g}%"
    print(f"\nHistorical volatility probe [{args.workspace}]")
    print(
        f"  anchor {anchor_date} | {args.volatility_node}={volatility:.4f} | "
        f"p{args.percentile:g} threshold={threshold:.4f} | {mode}"
    )

    probability, raw, deltas, horizons, observations, actual_as_of, fallback_cells = (
        build_historical_surface(workspace, data, features, anchor_date)
    )
    output_dir = ROOT / "probes" / args.workspace / f"high_vol_{actual_as_of}"
    workbook_path = output_dir / "bayes.xlsx"
    csv_path = output_dir / "expectancy.csv"
    summary_path = output_dir / "summary.json"
    workbooks.write_bayes_xlsx(
        workbook_path,
        probability,
        deltas,
        horizons,
        observations,
        workspace.horizon_unit,
    )

    rows = exhaust_surface(probability, deltas, horizons, args.cost_bps)
    ranked = sorted(rows, key=lambda row: row["expectancy_conservative"], reverse=True)
    _write_csv(csv_path, ranked)
    positive = {
        policy: sum(row[f"expectancy_{policy}"] > 0.0 for row in rows)
        for policy in ("conservative", "midpoint", "optimistic")
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps({
        "workspace": args.workspace,
        "anchor": actual_as_of,
        "anchor_selection": {
            "mode": mode,
            "volatility_node": args.volatility_node,
            "value": volatility,
            "percentile": args.percentile,
            "threshold": threshold,
            "min_history": args.min_history,
        },
        "design": {
            "candidate_nodes": len(features),
            "top_k_reselected_per_target": workspace.selection_top_k,
            "training_cutoff": "only outcomes whose full horizon ended by the anchor",
            "cost_bps": args.cost_bps,
            "timeout_return": 0.0,
            "competing_risk_assumption": "independent marginal TP and SL hitting times",
            "same_day_policies": ["conservative", "midpoint", "optimistic"],
            "coherence": "isotonic projection across nested barriers and horizons",
            "sparse_weight_fallback": "target historical base rate",
        },
        "surface": {
            "rows": len(deltas),
            "columns": len(horizons),
            "raw_cells_adjusted_for_coherence": int(np.sum(np.abs(probability - raw) > 1e-10)),
            "prior_fallback_cells": fallback_cells,
            "minimum_probability": float(probability.min()),
            "maximum_probability": float(probability.max()),
        },
        "expectancy": {
            "combinations": len(rows),
            "positive": positive,
            "top_conservative": ranked[:args.top],
        },
        "files": {"workbook": "bayes.xlsx", "expectancy": "expectancy.csv"},
    }, indent=2), encoding="utf-8")

    _print(rows, "conservative", args.top, args.workspace, actual_as_of)
    print(f"\n  wrote {workbook_path.relative_to(ROOT)}")
    print(f"  wrote {csv_path.relative_to(ROOT)}")
    print(f"  wrote {summary_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
