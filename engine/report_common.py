"""Shared artifact helpers for report figures."""

from __future__ import annotations

from catalog import find_node


def Δ_pct(v: float, step: float) -> str:
    """Barrier label: keep half-percent grids distinct, but avoid noisy .0% labels."""
    decimals = 1 if step < 0.01 else 0
    return format(float(v), f"+.{decimals}%")


def headline_node(ws, cleared: dict, universe: dict) -> dict | None:
    """Return the cleared sheet with the strongest selected skew score."""
    rows = [c for c in cleared.get("cleared", []) if c.get("best_cell")]
    if not rows:
        return None
    best = max(rows, key=lambda c: abs(c.get("selection_score") or c["best_cell"]["dev"]))
    node = find_node(universe, best["node"])
    return {**node, "cell": best["best_cell"], "gate": best}
