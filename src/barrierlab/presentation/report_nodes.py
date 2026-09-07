"""Single-node report figures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

from barrierlab.domain import barrier
from barrierlab.infrastructure import artifact_io
from barrierlab.presentation.report_style import (
    CMAP_DIV,
    GRID,
    INK,
    INK_2,
    MUTED,
    S2,
    SEQ,
    SURFACE,
    frame as _frame,
    note as _note,
    place_labels as _place_labels,
    plt,
    save as _save,
    title as _title,
)
from barrierlab.presentation.workbooks import feature_label

SHIFT_CMAP_LIMIT_PP = 30.0


def fig_band(ws, out: Path) -> Path | None:
    """
    The configured historical weighted-Bayes face, inverted onto its realized price path.

    A surface of touch probabilities is the honest object but not a legible one. Turned
    around -- *how far does price get, at what odds* -- and hung off the last close of a
    real price series, it becomes the thing the measurement was always about: a forward
    envelope, drawn on the chart it belongs to. Stage 6 fits this complete face using only
    outcomes completed by today's requested report date; Stage 7 only renders it.
    """
    if not ws.composition_array_path.exists():
        return None

    artifact = artifact_io.load_composition(ws.composition_array_path)
    needed = (
        "demonstration_probability",
        "demonstration_as_of",
        "current_Δ",
        "current_horizon",
        "current_node_ids",
    )
    if any(key not in artifact for key in needed):
        return None
    current_surface = artifact["demonstration_probability"].astype(float)
    as_of = str(np.asarray(artifact["demonstration_as_of"]).item())
    th = artifact["current_Δ"].astype(float)
    ts = artifact["current_horizon"].astype(int)
    n_nodes = len(artifact["current_node_ids"])

    if current_surface.shape != (len(th), len(ts)) or not as_of:
        return None
    if "report_index" not in artifact or "report_close" not in artifact:
        return None
    idx = pd.to_datetime(artifact["report_index"])
    price = pd.Series(artifact["report_close"].astype(float), index=idx).dropna()
    matches = np.flatnonzero(price.index == pd.Timestamp(as_of))
    if not len(matches):
        return None
    anchor_index = int(matches[-1])

    n_hist = int(min(len(price), max(40, 6 * int(ts.max()))))
    hist = price.iloc[max(0, anchor_index - n_hist + 1):anchor_index + 1]
    last = float(hist.iloc[-1])
    if anchor_index + int(ts.max()) < len(price):
        fwd = [price.index[anchor_index + int(t)] for t in ts]
    else:
        step = pd.Series(hist.index).diff().median()
        fwd = [hist.index[-1] + step * int(t) for t in ts]
    realized = price.iloc[
        anchor_index:min(len(price), anchor_index + int(ts.max()) + 1)
    ]

    bands = [
        (0.25, *barrier.touch_band(current_surface, th, 0.25), SEQ[2]),
        (0.50, *barrier.touch_band(current_surface, th, 0.50), SEQ[8]),
    ]

    fig, ax = plt.subplots(figsize=(9.6, 5.0))

    # history
    ax.plot(hist.index, hist.to_numpy(), color=INK, linewidth=1.4, zorder=6)

    # Lower touch probabilities imply farther barriers, so the wider 25% envelope goes
    # down first and the 50% band sits inside.
    for z, (q, up, dn, colour) in enumerate(bands, start=1):
        ax.fill_between(fwd, last * (1 - dn), last * (1 + up),
                        color=colour, alpha=0.16, linewidth=0, zorder=2 + z)
        for arm in (last * (1 + up), last * (1 - dn)):
            ax.plot(fwd, arm, color=SURFACE, linewidth=3.0, zorder=3 + z)
            ax.plot(fwd, arm, color=colour, linewidth=1.35, zorder=4 + z)

    if len(realized) > 1:
        ax.plot(realized.index, realized.to_numpy(), color=S2, linewidth=1.5, zorder=7)
    ax.axvline(hist.index[-1], color=MUTED, linewidth=0.9, zorder=2)
    ax.plot([hist.index[-1]], [last], marker='o', markersize=5.5, color=INK,
            markeredgecolor=SURFACE, markeredgewidth=1.4, zorder=8)

    ax.set_ylabel(f'{ws.asset["ticker"]} close')
    _frame(ax, grid_axis='y')

    j = len(ts) - 1
    ax.annotate(f'{last:,.0f}', (hist.index[-1], last), xytext=(-8, 11),
                textcoords='offset points', ha='right', fontsize=8.5,
                fontweight='bold', color=INK,
                bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1.2, "alpha": 0.9})

    end_labels = []
    for q, up, dn, colour in bands:
        end_labels += [
            (last * (1 + up[j]), f'{q:.0%} chance to hit this level  +{up[j]:.0%}', colour),
            (last * (1 - dn[j]), f'{q:.0%} chance to hit this level  -{dn[j]:.0%}', colour),
        ]
    _place_labels(ax, fwd[-1], [(y, text, colour) for y, text, colour in end_labels
                                if np.isfinite(y)])

    up_max = [float(np.nanmax(up)) for _, up, _, _ in bands if np.isfinite(up).any()]
    dn_max = [float(np.nanmax(dn)) for _, _, dn, _ in bands if np.isfinite(dn).any()]
    hi_y = max(float(hist.max()), float(realized.max()),
               last * (1 + max(up_max, default=0.0)))
    lo_y = min(float(hist.min()), float(realized.min()),
               last * (1 - max(dn_max, default=0.0)))
    pad = (hi_y - lo_y) * 0.07
    ax.set_ylim(lo_y - pad, hi_y + pad)

    _title(fig, f'Full Bayes probability band as of {hist.index[-1].date()}',
           f'The 25% and 50% touch envelopes combine all {n_nodes} validation-cleared '
           f'nodes in their states on that date. The orange line is the path realized '
           f'after the model cutoff.')
    _note(fig, f'{ws.asset["ticker"]} · +1..+{int(ts[j])}{ws.horizon_unit} · model fit '
               f'only with outcomes completed by {as_of} · coherent surface from '
               f'06_composition/composition.npz · one Stage 5 redundancy-cluster '
               f'representative per group, equal Naive-Bayes contribution · as-of date is the latest '
               f'available bar for today\'s report run')
    fig.subplots_adjust(top=0.78, right=0.78, bottom=0.10)
    return _save(fig, out / 'A_band.png')


def fig_full_bayes(ws, out: Path) -> Path | None:
    """Render the current combined Bayes shift from the unconditional surface."""
    if not ws.composition_array_path.exists():
        return None

    artifact = artifact_io.load_composition(ws.composition_array_path)
    needed = ('current_shift', 'current_Δ', 'current_horizon', 'current_node_ids')
    if any(key not in artifact for key in needed):
        return None

    shift = np.asarray(artifact['current_shift'], dtype=float)
    th = np.asarray(artifact['current_Δ'], dtype=float)
    ts = np.asarray(artifact['current_horizon'], dtype=int)
    n_nodes = len(artifact['current_node_ids'])
    as_of = str(np.asarray(artifact.get('current_as_of', '')).item())
    if shift.shape != (len(th), len(ts)) or not np.isfinite(shift).any():
        return None
    keep = np.abs(th) > 1e-12
    th = th[keep]
    shift = shift[keep]

    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    mesh = ax.pcolormesh(
        ts, th * 100, shift, cmap=CMAP_DIV,
        norm=TwoSlopeNorm(vcenter=0.0, vmin=-SHIFT_CMAP_LIMIT_PP,
                          vmax=SHIFT_CMAP_LIMIT_PP),
        shading='nearest',
    )
    ax.axhline(0, color=SURFACE, linewidth=1.4)

    cb = fig.colorbar(mesh, ax=ax, pad=0.02, fraction=0.04)
    cb.set_label('deviation from unconditional (%)', color=INK_2, fontsize=8)
    cb.outline.set_visible(False)
    cb.ax.tick_params(color=GRID, labelsize=7.5)

    ax.set_xlabel('horizon (+t days)')
    ax.set_ylabel('barrier Δ (%)')
    for side in ('top', 'right', 'left', 'bottom'):
        ax.spines[side].set_visible(False)

    _title(
        fig,
        'Full Bayes deviation from unconditional',
        f'Each cell combines all {n_nodes} cluster-representative, validation-cleared '
        f'conditions in their current states.',
    )
    _note(
        fig,
        f'{ws.asset["ticker"]} · as of {as_of} · coherent probability surface from '
        f'06_composition/composition.npz · equal-weight Naive-Bayes contribution per '
        f'Stage 5 redundancy cluster representative',
    )
    fig.subplots_adjust(top=0.80)
    return _save(fig, out / 'D_full_bayes.png')


def _render_shift(ws, head: dict, cube: dict, out_path: Path) -> Path | None:
    """Render one cleared node's strongest bin as a deviation from the baseline."""
    b = int(head['cell']['bin'])

    th, ts = cube['Δs'], cube['horizons']
    keep = np.abs(th) > 1e-12
    dev = cube['shift'][:, b, :][keep]
    th = th[keep]
    lim = SHIFT_CMAP_LIMIT_PP

    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    mesh = ax.pcolormesh(ts, th * 100, dev, cmap=CMAP_DIV,
                         norm=TwoSlopeNorm(vcenter=0.0, vmin=-lim, vmax=lim),
                         shading='nearest')
    ax.axhline(0, color=SURFACE, linewidth=1.4)

    cell = head['cell']
    ax.plot([cell['horizon']], [cell['Δ'] * 100], marker='o', markersize=7,
            markerfacecolor='none', markeredgecolor=INK, markeredgewidth=1.4)
    # a cell near the right edge would push its label under the colorbar
    right = cell['horizon'] > ts.min() + 0.7 * (ts.max() - ts.min())
    ax.annotate(f"{cell['dev']:+.1f}%", (cell['horizon'], cell['Δ'] * 100),
                xytext=(-10 if right else 10, 0), textcoords='offset points',
                fontsize=8, fontweight='bold', color=INK, va='center',
                ha='right' if right else 'left')

    cb = fig.colorbar(mesh, ax=ax, pad=0.02, fraction=0.04)
    cb.set_label('deviation from unconditional (%)', color=INK_2, fontsize=8)
    cb.outline.set_visible(False)
    cb.ax.tick_params(color=GRID, labelsize=7.5)

    ax.set_xlabel('horizon (+t days)')
    ax.set_ylabel('barrier Δ (%)')
    for side in ('top', 'right', 'left', 'bottom'):
        ax.spines[side].set_visible(False)

    condition = str(head["gate"].get("bin_label", "x")).replace("x", feature_label(head["id"]))
    _title(fig, f'Chance deviates by {abs(cell["dev"]):.1f}%, when {condition}',
           f'This is the conditional surface minus the baseline. Red means the barrier '
           f'is reached more often than usual; blue means less. The ring is the '
           f'strongest actionable cell.')
    _note(fig, f'{head["gate"].get("bin_label", "selected bin")} · ring at +{cell["horizon"]}'
               f'{ws.horizon_unit}, q = {head["gate"]["q_value"]:.2g} after '
               f'Benjamini-Hochberg · bin holds {cell["bin_n"]} bars · '
               f'06_composition/composition.npz report bundle')
    fig.subplots_adjust(top=0.80)
    return _save(fig, out_path)


def fig_shift_all(ws, cleared, out: Path) -> list[Path]:
    """Render one conditional-shift figure for every node that cleared all three filters."""
    paths: list[Path] = []
    artifact = artifact_io.load_composition(ws.composition_array_path)
    ids = [str(value) for value in artifact.get("report_node_ids", [])]
    for row in cleared.get('cleared', []):
        if not row.get('best_cell'):
            continue
        try:
            node = ws.catalog.find(row['node'])
            index = ids.index(node["id"])
            cube = {
                "Δs": artifact["current_Δ"],
                "horizons": artifact["current_horizon"],
                "shift": artifact[f"report_shift_{index}"],
            }
            head = {**node, 'cell': row['best_cell'], 'gate': row}
            suffix = f'{node["id"]}_bin{int(row.get("bin_number", row["best_cell"]["bin"] + 1))}'
            path = _render_shift(ws, head, cube, out / f'E_shift_{suffix}.png')
        except Exception:
            continue
        if path is not None:
            paths.append(path)
    return paths


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
            head = {**node, "cell": cell, "gate": {
                "bin_label": row.get("bin_label", f"bin {bin_index + 1}"),
                "q_value": float("nan"),
            }}
            paths.append(_render_shift(
                ws, head, cube,
                out / f'selected_shift_surface__{row["rank"]:03d}__{row["node"]}.png'
            ))
        except Exception:
            continue
    return [path for path in paths if path is not None]
