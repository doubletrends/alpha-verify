"""Stage 4 validation diagnostic figures."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from barrierlab.presentation.plot_style import (
    INK_2,
    S1,
    SURFACE,
    frame as _frame,
    note as _note,
    plt,
    save as _save,
    title as _title,
)
from barrierlab.presentation.workbooks import feature_label


def write_bin_score_null_histograms(ws, summary: dict, progress=None) -> list[Path]:
    """Render each selected-bin comparison against its simulated-bin null."""
    out = ws.validation_summary_path.parent / "plot"
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for row in summary.get("tests", []):
        try:
            null = np.asarray(row.get("null_scores", []), dtype=float)
            null = null[np.isfinite(null)]
            if not len(null):
                continue
            observed = float(row["bin_score"])
            p95 = float(row["null_p95"])
            # Freedman-Diaconis adapts to skewed, long-tailed null distributions;
            # bounds keep the result readable for both small and large ensembles.
            fd_edges = np.histogram_bin_edges(null, bins="fd")
            bins = min(80, max(24, len(fd_edges) - 1))
            fig, ax = plt.subplots(figsize=(7.2, 4.1))
            ax.hist(
                null,
                bins=bins,
                color="#d8d7d2",
                edgecolor=SURFACE,
                linewidth=0.6,
            )
            ax.axvline(
                p95,
                color=INK_2,
                linewidth=1.2,
                linestyle="--",
                label="raw p<0.05 threshold (null p95)",
            )
            ax.axvline(
                observed,
                color=S1,
                linewidth=2.1,
                label="observed bin score",
            )
            ax.set_xlabel("two-sided bin score (pp)")
            ax.set_ylabel("synthetic OHLC paths")
            _frame(ax, grid_axis="y")
            ax.legend(fontsize=8, frameon=False)
            _title(
                fig,
                f'{feature_label(row["node"])} bin {int(row["bin_number"])} — score vs synthetic null',
                f'observed {observed:.2f}pp · null p95 {p95:.2f}pp · '
                f'p={row["peak_p"]:.4f} · {"CLEARED" if row.get("cleared") else "NOT CLEARED"}',
            )
            _note(
                fig,
                f'{ws.dir.name} · {len(null)} shared synthetic OHLC paths · '
                "external condition histories held fixed",
            )
            fig.subplots_adjust(top=0.78, bottom=0.18)
            path = (
                out
                / f'null_histogram__{row["rank"]:03d}__{row["node"]}__bin_{int(row["bin_number"]):02d}.png'
            )
            paths.append(_save(fig, path))
        finally:
            if progress is not None:
                progress.advance()
    return paths
