"""Probe bracket-trade expectancy from a Stage 6 current probability surface.

This is deliberately an exploratory read-only tool, not a pipeline stage or backtest.
Stage 6 stores marginal cumulative touch probabilities, not observed joint first-hit
paths.  The probe therefore treats the take-profit and stop-loss hitting times as
independent competing risks, derives their daily first-hit masses, and reports three
same-day ordering assumptions.

Examples:
    python scripts/probe_expectancy.py --workspace nasdaq_daily
    python scripts/probe_expectancy.py --workspace btc_daily --cost-bps 15 --top 30
    python scripts/probe_expectancy.py --workspace btc_daily --csv btc_probe.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def first_hit_probabilities(
    take_profit_cdf: np.ndarray,
    stop_loss_cdf: np.ndarray,
) -> tuple[float, float, float, float]:
    """Return strict TP-first, strict SL-first, same-day tie, and timeout masses."""
    tp = np.maximum.accumulate(np.clip(np.asarray(take_profit_cdf, float), 0.0, 1.0))
    sl = np.maximum.accumulate(np.clip(np.asarray(stop_loss_cdf, float), 0.0, 1.0))
    if tp.shape != sl.shape or tp.ndim != 1 or not len(tp):
        raise ValueError("TP and SL curves must be non-empty one-dimensional peers")

    tp_mass = np.diff(np.r_[0.0, tp])
    sl_mass = np.diff(np.r_[0.0, sl])
    tp_first = float(np.sum(tp_mass * (1.0 - sl)))
    sl_first = float(np.sum(sl_mass * (1.0 - tp)))
    same_day = float(np.sum(tp_mass * sl_mass))
    timeout = float((1.0 - tp[-1]) * (1.0 - sl[-1]))

    probabilities = np.array([tp_first, sl_first, same_day, timeout], dtype=float)
    probabilities = np.clip(probabilities, 0.0, 1.0)
    total = float(probabilities.sum())
    if total <= 0.0:
        raise ValueError("competing-risk probabilities have zero mass")
    probabilities /= total
    return tuple(float(value) for value in probabilities)


def exhaust_surface(
    probability: np.ndarray,
    deltas: np.ndarray,
    horizons: np.ndarray,
    cost_bps: float = 0.0,
) -> list[dict]:
    """Evaluate every positive TP, negative SL, and available timeout combination."""
    probability = np.asarray(probability, dtype=float)
    deltas = np.asarray(deltas, dtype=float)
    horizons = np.asarray(horizons, dtype=int)
    if probability.shape != (len(deltas), len(horizons)):
        raise ValueError("Stage 6 probability must be shaped delta x horizon")
    if not np.isfinite(probability).all():
        raise ValueError("Stage 6 probability contains missing values")
    if cost_bps < 0:
        raise ValueError("cost-bps cannot be negative")

    tp_indexes = np.flatnonzero(deltas > 0)
    sl_indexes = np.flatnonzero(deltas < 0)
    cost = float(cost_bps) / 10_000.0
    rows = []
    for tp_index in tp_indexes:
        take_profit = float(deltas[tp_index])
        for sl_index in sl_indexes:
            stop_loss = float(deltas[sl_index])
            for horizon_index, timeout_days in enumerate(horizons):
                tp_first, sl_first, same_day, timeout = first_hit_probabilities(
                    probability[tp_index, :horizon_index + 1],
                    probability[sl_index, :horizon_index + 1],
                )
                conservative = (
                    tp_first * take_profit
                    + (sl_first + same_day) * stop_loss
                    - cost
                )
                midpoint = (
                    tp_first * take_profit
                    + sl_first * stop_loss
                    + same_day * (take_profit + stop_loss) / 2.0
                    - cost
                )
                optimistic = (
                    (tp_first + same_day) * take_profit
                    + sl_first * stop_loss
                    - cost
                )
                rows.append({
                    "take_profit": take_profit,
                    "stop_loss": stop_loss,
                    "timeout": int(timeout_days),
                    "expectancy_conservative": float(conservative),
                    "expectancy_midpoint": float(midpoint),
                    "expectancy_optimistic": float(optimistic),
                    "p_tp_first": tp_first,
                    "p_sl_first": sl_first,
                    "p_same_day": same_day,
                    "p_timeout": timeout,
                    "cost_bps": float(cost_bps),
                })
    return rows


def _percent(value: float) -> str:
    return f"{value:>8.2%}"


def _print(rows: list[dict], rank_by: str, top: int, workspace: str, as_of: str) -> None:
    key = f"expectancy_{rank_by}"
    ranked = sorted(rows, key=lambda row: row[key], reverse=True)
    positive = {
        policy: sum(row[f"expectancy_{policy}"] > 0.0 for row in rows)
        for policy in ("conservative", "midpoint", "optimistic")
    }
    cost_bps = rows[0]["cost_bps"] if rows else 0.0

    print(f"\nStage 6 expectancy probe [{workspace}]  as of {as_of}")
    print(
        f"  {len(rows):,} combinations | cost {cost_bps:g} bps round trip | "
        "timeout return = 0"
    )
    print("  approximation: independent TP/SL hitting times from marginal Stage 6 curves")
    print(
        "  positive combinations: "
        f"conservative {positive['conservative']:,} | "
        f"midpoint {positive['midpoint']:,} | "
        f"optimistic {positive['optimistic']:,}"
    )
    print(f"\n  top {min(top, len(ranked))} ranked by {rank_by} expectancy")
    print(
        "    TP      SL   days    E[cons]    E[mid]    E[opt]  "
        "TP first  SL first  same day   timeout"
    )
    print("  " + "-" * 100)
    for row in ranked[:top]:
        print(
            f"  {_percent(row['take_profit'])} {_percent(row['stop_loss'])} "
            f"{row['timeout']:>5} {_percent(row['expectancy_conservative'])} "
            f"{_percent(row['expectancy_midpoint'])} "
            f"{_percent(row['expectancy_optimistic'])} "
            f"{_percent(row['p_tp_first'])} {_percent(row['p_sl_first'])} "
            f"{_percent(row['p_same_day'])} {_percent(row['p_timeout'])}"
        )


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exhaust bracket expectancies from a Stage 6 probability artifact."
    )
    parser.add_argument("--workspace", default="nasdaq_daily")
    parser.add_argument("--cost-bps", type=float, default=0.0, metavar="BPS")
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument(
        "--rank-by",
        choices=("conservative", "midpoint", "optimistic"),
        default="conservative",
    )
    parser.add_argument("--csv", type=Path, help="optionally save all combinations")
    args = parser.parse_args()

    artifact_path = (
        ROOT / "workspaces" / args.workspace / "06_bayes" / "bayes.npz"
    )
    if not artifact_path.exists():
        raise SystemExit(f"Missing {artifact_path}; run Stage 6 first.")
    with np.load(artifact_path, allow_pickle=False) as artifact:
        required = {"current_probability", "current_Δ", "current_horizon", "current_as_of"}
        missing = sorted(required - set(artifact.files))
        if missing:
            raise SystemExit(f"Stage 6 artifact lacks: {', '.join(missing)}")
        rows = exhaust_surface(
            artifact["current_probability"],
            artifact["current_Δ"],
            artifact["current_horizon"],
            args.cost_bps,
        )
        as_of = str(artifact["current_as_of"])

    if args.top < 1:
        raise SystemExit("--top must be at least 1")
    _print(rows, args.rank_by, args.top, args.workspace, as_of)
    if args.csv:
        _write_csv(args.csv, rows)
        print(f"\n  wrote {args.csv.resolve()}")


if __name__ == "__main__":
    main()
