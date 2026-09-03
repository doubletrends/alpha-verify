"""Shared artifact helpers for report figures."""

from __future__ import annotations

from engine import barrier
from tree.tree import find_node
from workspace import BASELINE_NODE


def theta_pct(v: float, step: float) -> str:
    """Barrier label: keep half-percent grids distinct, but avoid noisy .0% labels."""
    decimals = 1 if step < 0.01 else 0
    return format(float(v), f"+.{decimals}%")


def full_baseline(ws) -> dict:
    """
    The baseline node's full cube, not its summary.

    The report renders against the full grid because that is the faithful record and
    draws a smooth cone. The summary grid exists to be judged on.
    """
    return barrier.load_cube(ws.cube_path("_base", BASELINE_NODE))


def headline_node(ws, cleared: dict, tree: dict) -> dict | None:
    """Return the cleared node whose best cell moves the barrier rate furthest."""
    rows = [c for c in cleared.get("cleared", []) if c.get("best_cell")]
    if not rows:
        return None
    best = max(rows, key=lambda c: abs(c["best_cell"]["dev"]))
    node = find_node(tree, best["node"])
    return {**node, "cell": best["best_cell"], "gate": best}
