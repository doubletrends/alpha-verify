"""Stage 4 full-grid heatmaps for statistically cleared condition bins."""

from __future__ import annotations

from pathlib import Path
import shutil

import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import PercentFormatter

from barrierlab.presentation.plot_style import (
    CMAP_DIV, GRID, INK_2, SURFACE,
    note as _note, plt, save as _save, title as _title,
)
from barrierlab.presentation.workbooks import feature_label

SHIFT_LIMIT = 0.30


def _condition(row: dict) -> str:
    return str(row.get("bin_label", f'bin {int(row["bin_number"])}')).replace(
        "x", feature_label(row["node"])
    )


def write_selected_shift_heatmaps(
    ws, selected: list[dict], cubes: dict[str, dict], progress=None,
) -> list[Path]:
    """Render the complete delta-by-horizon shift surface for each selected bin."""
    out = ws.selection_summary_path.parent / "plot"
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for row in selected:
        try:
            cube = cubes[row["node"]]
            bin_index = int(row["bin"])
            barriers = np.asarray(cube["barriers"], dtype=float)
            horizons = np.asarray(cube["horizons"], dtype=int)
            keep = np.abs(barriers) > 1e-12
            surface = np.asarray(
                cube["probability_shift"][:, bin_index, :], dtype=float
            )[keep]
            shown_barriers = barriers[keep]

            fig, ax = plt.subplots(figsize=(8.4, 5.0))
            mesh = ax.pcolormesh(
                horizons, shown_barriers * 100.0, surface,
                cmap=CMAP_DIV,
                norm=TwoSlopeNorm(vcenter=0.0, vmin=-SHIFT_LIMIT, vmax=SHIFT_LIMIT),
                shading="nearest",
            )
            ax.axhline(0, color=SURFACE, linewidth=1.4)
            colorbar = fig.colorbar(mesh, ax=ax, pad=0.02, fraction=0.04)
            colorbar.set_label("conditional minus baseline (probability difference)", color=INK_2)
            colorbar.formatter = PercentFormatter(xmax=1.0)
            colorbar.update_ticks()
            colorbar.outline.set_visible(False)
            colorbar.ax.tick_params(color=GRID, labelsize=7.5)
            ax.set_xlabel(f"forward horizon (+t {ws.horizon_unit})")
            ax.set_ylabel("signed barrier delta (%)")
            for side in ("top", "right", "left", "bottom"):
                ax.spines[side].set_visible(False)
            _title(
                fig,
                f'{feature_label(row["node"])} bin {int(row["bin_number"])}: {_condition(row)}',
                f'full baseline-relative surface · validation score {float(row["bin_score"]):.4f} '
                f'· raw p={float(row["monte_carlo_p_value"]):.4f}',
            )
            _note(
                fig,
                "Red: barrier touched more often than the unconditional market baseline. "
                "Blue: touched less often. Stage 4 includes only validation-cleared bins.",
            )
            fig.subplots_adjust(top=0.80)
            path = out / (
                f'selected_shift_heatmap__{row["node"]}__'
                f'bin_{int(row["bin_number"]):02d}.png'
            )
            paths.append(_save(fig, path))
        finally:
            if progress is not None:
                progress.advance()

    current = set(paths)
    for path in out.glob("selected_shift_heatmap__*.png"):
        if path not in current:
            path.unlink()
    return paths


def copy_selected_null_histograms(ws, selected: list[dict]) -> tuple[list[Path], list[str]]:
    """Copy Stage 3 null distributions beside the selected-bin heatmaps."""
    source_dir = ws.validation_summary_path.parent / "plot"
    out = ws.selection_summary_path.parent / "plot"
    out.mkdir(parents=True, exist_ok=True)
    copied, missing = [], []
    for row in selected:
        name = f'null_histogram__{row["node"]}__bin_{int(row["bin_number"]):02d}.png'
        source = source_dir / name
        target = out / name
        if not source.exists():
            missing.append(name)
            continue
        shutil.copy2(source, target)
        copied.append(target)

    current = set(copied)
    for path in out.glob("null_histogram__*.png"):
        if path not in current:
            path.unlink()
    return copied, missing
