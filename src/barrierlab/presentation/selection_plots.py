"""Stage 3 selected-node figures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.colors import TwoSlopeNorm

from barrierlab.infrastructure import artifact_io
from barrierlab.presentation.plot_style import (
    CMAP_DIV,
    GRID,
    INK,
    INK_2,
    SURFACE,
    note as _note,
    plt,
    save as _save,
    title as _title,
)
from barrierlab.presentation.workbooks import feature_label

SHIFT_CMAP_LIMIT_PP = 30.0


def _render_shift(ws, head: dict, cube: dict, out_path: Path) -> Path | None:
    """Render one selected node's strongest bin as a deviation from the baseline."""
    b = int(head["cell"]["bin"])

    th, ts = cube["Δs"], cube["horizons"]
    keep = np.abs(th) > 1e-12
    dev = cube["shift"][:, b, :][keep]
    th = th[keep]

    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    mesh = ax.pcolormesh(
        ts,
        th * 100,
        dev,
        cmap=CMAP_DIV,
        norm=TwoSlopeNorm(
            vcenter=0.0,
            vmin=-SHIFT_CMAP_LIMIT_PP,
            vmax=SHIFT_CMAP_LIMIT_PP,
        ),
        shading="nearest",
    )
    ax.axhline(0, color=SURFACE, linewidth=1.4)

    cell = head["cell"]
    ax.plot(
        [cell["horizon"]],
        [cell["Δ"] * 100],
        marker="o",
        markersize=7,
        markerfacecolor="none",
        markeredgecolor=INK,
        markeredgewidth=1.4,
    )
    right = cell["horizon"] > ts.min() + 0.7 * (ts.max() - ts.min())
    ax.annotate(
        f'{cell["dev"]:+.1f}%',
        (cell["horizon"], cell["Δ"] * 100),
        xytext=(-10 if right else 10, 0),
        textcoords="offset points",
        fontsize=8,
        fontweight="bold",
        color=INK,
        va="center",
        ha="right" if right else "left",
    )

    cb = fig.colorbar(mesh, ax=ax, pad=0.02, fraction=0.04)
    cb.set_label("deviation from unconditional (%)", color=INK_2, fontsize=8)
    cb.outline.set_visible(False)
    cb.ax.tick_params(color=GRID, labelsize=7.5)

    ax.set_xlabel("horizon (+t days)")
    ax.set_ylabel("barrier Δ (%)")
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)

    condition = str(head["gate"].get("bin_label", "x")).replace(
        "x", feature_label(head["id"])
    )
    _title(
        fig,
        f'Chance deviates by {abs(cell["dev"]):.1f}%, when {condition}',
        "This is the conditional surface minus the baseline. Red means the barrier "
        "is reached more often than usual; blue means less. The ring is the "
        "strongest selected cell.",
    )
    _note(
        fig,
        f'{head["gate"].get("bin_label", "selected bin")} · ring at '
        f'+{cell["horizon"]}{ws.horizon_unit} · bin holds {cell["bin_n"]} bars · '
        "03_selection selected-node artifact",
    )
    fig.subplots_adjust(top=0.80)
    return _save(fig, out_path)


def write_selected_shift_graphs(ws, selected: list[dict]) -> list[Path]:
    """Write Stage 3 shift surfaces for the selected condition bins."""
    out = ws.selection_path.parent / "plot"
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for row in selected:
        try:
            cube = artifact_io.load_selected_node(ws.selection_array_path(row))
            bin_index = int(row["bin"])
            surface = np.asarray(cube["shift"][:, bin_index, :], dtype=float)
            flat = int(np.nanargmax(np.abs(surface)))
            delta_index, horizon_index = np.unravel_index(flat, surface.shape)
            cell = {
                "bin": bin_index,
                "Δ": float(cube["Δs"][delta_index]),
                "horizon": int(cube["horizons"][horizon_index]),
                "dev": float(surface[delta_index, horizon_index]),
                "bin_n": int(cube["bin_n"][bin_index, horizon_index]),
            }
            node = ws.catalog.find(row["node"])
            head = {
                **node,
                "cell": cell,
                "gate": {
                    "bin_label": row.get("bin_label", f"bin {bin_index + 1}"),
                },
            }
            paths.append(
                _render_shift(
                    ws,
                    head,
                    cube,
                    out
                    / f'selected_shift_surface__{row["rank"]:03d}__{row["node"]}.png',
                )
            )
        except Exception:
            continue
    return [path for path in paths if path is not None]
