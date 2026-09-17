"""Standard condition-bin figure: shift heatmap beside the bin's null distribution.

Stage 3 writes one per tested bin and Stage 4 one per selected bin; both use
this module so a bin looks the same wherever it is reviewed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import PercentFormatter

from alphaverify.presentation.display import SHIFT_DISPLAY_LIMIT, feature_label
from alphaverify.presentation.plot_style import (
    CMAP_DIV, GRID, INK, INK_2, S1, SURFACE,
    frame as _frame, note as _note, plt, save as _save, title as _title,
)


def finite_null_scores(row: dict) -> np.ndarray:
    """A validation row's synthetic null scores, without non-finite draws."""
    null = np.asarray(row.get("null_scores", []), dtype=float)
    return null[np.isfinite(null)]


def figure_name(row: dict) -> str:
    return f'bin_figure__{row["node"]}__bin_{int(row["bin_number"]):02d}.png'


def _condition(row: dict) -> str:
    return str(row.get("bin_label", f'bin {int(row["bin_number"])}')).replace(
        "x", feature_label(row["node"])
    )


def _panel_title(ax, text: str) -> None:
    ax.set_title(text, loc="left", fontsize=9.5, fontweight="bold", color=INK, pad=10)


def draw_shift_heatmap(fig, ax, cube: dict, bin_index: int, horizon_unit: str) -> None:
    """Draw one bin's complete signed-barrier-by-horizon shift surface and its colorbar."""
    barriers = np.asarray(cube["barriers"], dtype=float)
    horizons = np.asarray(cube["horizons"], dtype=int)
    keep = np.abs(barriers) > 1e-12
    surface = np.asarray(cube["probability_shift"][:, bin_index, :], dtype=float)[keep]

    mesh = ax.pcolormesh(
        horizons, barriers[keep] * 100.0, surface,
        cmap=CMAP_DIV,
        norm=TwoSlopeNorm(vcenter=0.0, vmin=-SHIFT_DISPLAY_LIMIT, vmax=SHIFT_DISPLAY_LIMIT),
        shading="nearest",
    )
    ax.axhline(0, color=SURFACE, linewidth=1.4)
    # The colorbar takes its width from this panel only, preserving the half-width split.
    colorbar = fig.colorbar(mesh, ax=ax, pad=0.02, fraction=0.05)
    colorbar.set_label("conditional minus baseline (probability difference)", color=INK_2)
    colorbar.formatter = PercentFormatter(xmax=1.0)
    colorbar.update_ticks()
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(color=GRID, labelsize=7.5)
    ax.set_xlabel(f"forward horizon (+t {horizon_unit})")
    ax.set_ylabel("signed barrier delta (%)")
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)


def draw_null_histogram(ax, row: dict, null: np.ndarray) -> None:
    """Draw one condition bin's observed score against its simulated-bin null."""
    observed = float(row["bin_score"])
    p95 = float(row["null_p95"])
    # Freedman-Diaconis adapts to skewed, long-tailed null distributions;
    # bounds keep the result readable for both small and large ensembles.
    fd_edges = np.histogram_bin_edges(null, bins="fd")
    bins = min(80, max(24, len(fd_edges) - 1))
    ax.hist(null, bins=bins, color="#d8d7d2", edgecolor=SURFACE, linewidth=0.6)
    ax.axvline(p95, color=INK_2, linewidth=1.2, linestyle="--", label="null 95th percentile")
    ax.axvline(observed, color=S1, linewidth=2.1, label="observed bin score")
    ax.set_xlabel("two-sided bin score (weighted probability difference)")
    ax.set_ylabel("synthetic OHLC replicates")
    _frame(ax, grid_axis="y")
    ax.legend(fontsize=8, frameon=False)


def write_bin_figures(
    out: Path, rows: list[dict], cubes: dict[str, dict], *,
    workspace: str, horizon_unit: str, progress=None,
) -> list[Path]:
    """Write one standard figure per validation row into ``out``.

    Rows are Stage 3 test records, including ``null_scores``; ``cubes`` maps
    each node to its materialized Stage 2 shift. The two panels share the
    figure width equally so the effect and its statistical evidence read
    together. The caller's stage owns ``out``: figures for rows no longer
    present are removed.
    """
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for row in rows:
        try:
            null = finite_null_scores(row)
            if not len(null):
                continue
            fig, (heatmap_ax, null_ax) = plt.subplots(
                1, 2, figsize=(14.4, 5.4),
                gridspec_kw={"width_ratios": (1, 1), "wspace": 0.22},
            )
            draw_shift_heatmap(fig, heatmap_ax, cubes[row["node"]], int(row["bin"]), horizon_unit)
            _panel_title(heatmap_ax, "Shift from the market baseline")
            draw_null_histogram(null_ax, row, null)
            _panel_title(null_ax, "Bin score vs synthetic null")
            _title(
                fig,
                f'{feature_label(row["node"])} bin {int(row["bin_number"])}: {_condition(row)}',
                f'validation score {float(row["bin_score"]):.4f} · '
                f'null p95 {float(row["null_p95"]):.4f} · '
                f'raw p={float(row["monte_carlo_p_value"]):.4f} · '
                f'{"CLEARED" if row.get("cleared") else "NOT CLEARED"}',
            )
            _note(
                fig,
                f"{workspace} · Left: red means the barrier is touched more often than the "
                "unconditional market baseline; blue, less often. "
                f"Right: {len(null)} synthetic OHLC replicates with external condition "
                "histories held fixed.",
            )
            fig.subplots_adjust(top=0.80, bottom=0.11, left=0.05, right=0.98)
            paths.append(_save(fig, out / figure_name(row)))
        finally:
            if progress is not None:
                progress.advance()

    current = set(paths)
    for path in out.glob("*.png"):
        if path not in current:
            path.unlink()
    return paths
