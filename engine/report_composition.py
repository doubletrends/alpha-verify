"""Composition report figures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.colors import TwoSlopeNorm

from engine.report_common import Δ_pct
from engine.report_style import (
    CMAP_DIV,
    GRID,
    INK_2,
    SURFACE,
    note,
    plt,
    save,
    title,
)


def fig_composition_grid(ws, out: Path) -> Path | None:
    """Where the composed model ranks, over the grid everything else is judged on."""
    if not ws.bayes_path.exists():
        return None
    z = np.load(ws.bayes_path, allow_pickle=False)
    th_all, hz_all, auc = z["grid_Δ"], z["grid_horizon"], z["grid_auc"]
    if not len(th_all):
        return None

    min_events = 10
    n_pos = z["grid_realized"] * z["grid_n_scored"]
    n_neg = z["grid_n_scored"] - n_pos
    usable = (n_pos >= min_events) & (n_neg >= min_events)

    th = np.array(sorted(set(th_all)))
    ts = np.array(sorted(set(hz_all)))
    m = np.full((len(th), len(ts)), np.nan)
    for t, t, a, ok in zip(th_all, hz_all, auc, usable):
        if ok:
            m[int(np.flatnonzero(th == t)[0]), int(np.flatnonzero(ts == t)[0])] = a
    if not np.isfinite(m).any():
        return None
    lim = float(np.nanmax(np.abs(m - 0.5)))

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    cmap = CMAP_DIV.copy()
    cmap.set_bad(SURFACE)
    mesh = ax.pcolormesh(
        np.arange(len(ts) + 1),
        np.arange(len(th) + 1),
        m,
        cmap=cmap,
        norm=TwoSlopeNorm(vcenter=0.5, vmin=0.5 - lim, vmax=0.5 + lim),
    )
    cb = fig.colorbar(mesh, ax=ax, pad=0.02, fraction=0.045)
    cb.set_label("out-of-sample AUC  (0.5 = no ranking)", color=INK_2, fontsize=8)
    cb.outline.set_visible(False)
    cb.ax.tick_params(color=GRID, labelsize=7.5)

    ax.set_xticks(np.arange(len(ts)) + 0.5)
    ax.set_xticklabels([f"+{t}{ws.horizon_unit}" for t in ts])
    ax.set_yticks(np.arange(len(th)) + 0.5)
    ax.set_yticklabels([Δ_pct(t, ws.Δ_step) for t in th], fontsize=7.5)
    ax.tick_params(length=0)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel(f"horizon (+{ws.horizon_unit})")
    ax.set_ylabel("barrier Δ")

    above = int(np.nansum(m > 0.5))
    tot = int(np.isfinite(m).sum())
    k = int(np.nanargmax(m))
    bi, bj = divmod(k, len(ts))
    title(
        fig,
        "Does the ranking survive, and where?",
        f"The same walk forward run at every target on the judged grid. Red ranks "
        f"better than chance out of sample, blue worse.\n"
        f"{above} of {tot} targets come out above 0.5. The strongest is "
        f"{Δ_pct(th[bi], ws.Δ_step)} within +{int(ts[bj])}{ws.horizon_unit} at AUC "
        f"{m[bi, bj]:.2f} — modest, and that is what an honest out-of-sample number "
        f"on this question looks like.",
    )
    note(
        fig,
        f"{ws.dir.name} · one-node-per-family model, everything fit on training "
        f"windows only, scored on non-overlapping bars · blank cells had fewer "
        f"than {min_events} events on one side, where AUC turns on two or three "
        f"cases · AUC is unchanged by the scale correction, which moves "
        f"calibration and not order · 06_bayes.npz",
    )
    fig.subplots_adjust(top=0.82)
    return save(fig, out / "10_composition_grid.png")
