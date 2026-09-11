"""Shared visual system for pipeline figures."""

from __future__ import annotations

import textwrap
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8983"
GRID = "#e6e5e1"
S1 = "#2a78d6"

CMAP_DIV = LinearSegmentedColormap.from_list(
    "div",
    [
        "#0d366b", "#1c5cab", "#2a78d6", "#86b6ef", "#cde2fb",
        "#f0efec",
        "#fbd9d8", "#f5aeac", "#ec7d7b", "#e34948", "#b62f2e",
    ],
)

plt.rcParams.update({
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
    "text.color": INK,
    "axes.labelcolor": INK_2,
    "axes.edgecolor": GRID,
    "axes.linewidth": 0.8,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.frameon": False,
    "axes.grid": False,
})


def frame(ax, grid_axis: str | None = "y") -> None:
    """Apply the pipeline figures' minimal axis chrome."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    if grid_axis:
        ax.set_axisbelow(True)
        ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.6, linestyle="-")


def wrap(text: str, width: int) -> str:
    """Hard-wrap each already-broken line to ``width`` characters."""
    return "\n".join(
        textwrap.fill(line, width) if line else "" for line in text.split("\n")
    )


def title(fig, title_text: str, subtitle: str, y: float = 0.98) -> None:
    fig.text(
        0.012,
        y,
        wrap(title_text, 82),
        ha="left",
        va="top",
        fontsize=12.5,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.012,
        y - 0.055,
        wrap(subtitle, 108),
        ha="left",
        va="top",
        fontsize=8.5,
        color=INK_2,
    )


def note(fig, text: str) -> None:
    fig.text(
        0.012,
        -0.3 / fig.get_figheight(),
        wrap(text, 132),
        ha="left",
        va="top",
        fontsize=7.2,
        color=MUTED,
    )


def save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        for attempt in range(3):
            try:
                fig.savefig(path, bbox_inches="tight", pad_inches=0.28)
                break
            except OSError:
                if attempt == 2:
                    raise
                time.sleep(0.2)
    finally:
        plt.close(fig)
    return path
