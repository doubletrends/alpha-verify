"""Single-node report figures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

from engine import barrier, shift
from engine.report_common import headline_node as _headline_node
from engine.report_common import Δ_pct as _Δ_pct
from engine.report_style import (
    CMAP_DIV,
    CMAP_SEQ,
    FAINT,
    GRID,
    INK,
    INK_2,
    MUTED,
    S1,
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
from engine.writer import feature_label
from catalog import find_node

SHIFT_CMAP_LIMIT_PP = 30.0


def fig_band(ws, universe, cleared, out: Path) -> Path | None:
    """
    The cube, inverted and anchored to the price it describes.

    A surface of touch probabilities is the honest object but not a legible one. Turned
    around -- *how far does price get, at what odds* -- and hung off the last close of a
    real price series, it becomes the thing the measurement was always about: a forward
    envelope, drawn on the chart it belongs to.

    The condition is read from the last bar, so the band shown is the one that applies to
    the regime the asset is actually in. That is emphatically not a forecast: it is the
    historical frequency with which price reached each level under the decile the feature
    currently sits in, and nothing in it knows anything about the future.
    """
    head = _headline_node(ws, cleared, universe)
    if head is None:
        return None
    cube = shift.load(ws.shift_cube_path(head['family'], head['id']))
    needed = ("index", "feature_values", "close")
    if any(k not in cube for k in needed):
        return None
    prob, th, ts = cube['prob'], cube['Δs'], cube['horizons']
    n_bins = prob.shape[1]
    if n_bins < 2:
        return None

    idx = pd.to_datetime(cube["index"])
    panel = pd.DataFrame(
        {"price": cube["close"].astype(float), "feature": cube["feature_values"].astype(float)},
        index=idx,
    ).dropna()
    if panel.empty:
        return None
    now_val = float(panel["feature"].iloc[-1])
    now_bin = int(np.searchsorted(cube['edges'], now_val)) if len(cube['edges']) else 0
    now_bin = int(np.clip(now_bin, 0, n_bins - 1))

    price = panel["price"]
    n_hist = int(min(len(price), max(40, 6 * int(ts.max()))))
    hist = price.iloc[-n_hist:]
    last = float(hist.iloc[-1])
    step = pd.Series(hist.index).diff().median()
    fwd = [hist.index[-1] + step * int(t) for t in ts]

    labels = cube['meta']['bin_labels']
    current_surface = prob[:, now_bin, :]
    bands = [
        (0.25, *barrier.touch_band(current_surface, th, 0.25), SEQ[2]),
        (0.50, *barrier.touch_band(current_surface, th, 0.50), SEQ[8]),
    ]

    fig, ax = plt.subplots(figsize=(9.6, 5.0))

    # history
    ax.plot(hist.index, hist.to_numpy(), color=INK, linewidth=1.4, zorder=6)

    # Draw only the current condition bin. Lower touch probabilities imply farther
    # barriers, so the wider 25% envelope goes down first and the 50% band sits inside.
    for z, (q, up, dn, colour) in enumerate(bands, start=1):
        ax.fill_between(fwd, last * (1 - dn), last * (1 + up),
                        color=colour, alpha=0.16, linewidth=0, zorder=2 + z)
        for arm in (last * (1 + up), last * (1 - dn)):
            ax.plot(fwd, arm, color=SURFACE, linewidth=3.0, zorder=3 + z)
            ax.plot(fwd, arm, color=colour, linewidth=1.35, zorder=4 + z)

    ax.axvline(hist.index[-1], color=MUTED, linewidth=0.9, zorder=2)
    ax.plot([hist.index[-1]], [last], marker='o', markersize=5.5, color=INK,
            markeredgecolor=SURFACE, markeredgewidth=1.4, zorder=8)

    ax.set_ylabel(f'{ws.asset["ticker"]} close')
    _frame(ax, grid_axis='y')

    j = len(ts) - 1
    ax.annotate(f'{last:,.0f}', (hist.index[-1], last), xytext=(-8, 11),
                textcoords='offset points', ha='right', fontsize=8.5,
                fontweight='bold', color=INK)

    end_labels = []
    for q, up, dn, colour in bands:
        end_labels += [
            (last * (1 + up[j]), f'{q:.0%} touch  +{up[j]:.0%}   {last * (1 + up[j]):,.0f}', colour),
            (last * (1 - dn[j]), f'{q:.0%} touch  -{dn[j]:.0%}   {last * (1 - dn[j]):,.0f}', colour),
        ]
    _place_labels(ax, fwd[-1], [(y, text, colour) for y, text, colour in end_labels
                                if np.isfinite(y)])

    up_max = [float(np.nanmax(up)) for _, up, _, _ in bands if np.isfinite(up).any()]
    dn_max = [float(np.nanmax(dn)) for _, _, dn, _ in bands if np.isfinite(dn).any()]
    hi_y = max(float(hist.max()), last * (1 + max(up_max, default=0.0)))
    lo_y = min(float(hist.min()), last * (1 - max(dn_max, default=0.0)))
    pad = (hi_y - lo_y) * 0.07
    ax.set_ylim(lo_y - pad, hi_y + pad)

    _title(fig, f'Probability forecast demonstration using today\'s {feature_label(head["id"])} condition',
           f'The bands are the 25% and 50% historical touch envelopes when '
           f'{feature_label(head["id"])} sits in today\'s bin: {labels[now_bin]}. '
           f'Each envelope is measured from the same current condition bin.')
    _note(fig, f'{ws.asset["ticker"]} through {hist.index[-1].date()} · not a forecast · '
               f'25% / 50% touch probabilities, inverted from 02_shift_array/{head["family"]}/'
               f'{head["id"]}.npz over +1..+{int(ts[j])}{ws.horizon_unit} · current bin '
               f'holds {int(cube["bin_n"][now_bin, j])} bars')
    fig.subplots_adjust(top=0.78, right=0.78, bottom=0.10)
    return _save(fig, out / 'A_band.png')


def _render_shift(ws, head: dict, out_path: Path) -> Path | None:
    """Render one cleared node's strongest bin as a deviation from the baseline."""
    cube = shift.load(ws.shift_cube_path(head['family'], head['id']))
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
    ax.annotate(f"{cell['dev']:+.1f} pp", (cell['horizon'], cell['Δ'] * 100),
                xytext=(-10 if right else 10, 0), textcoords='offset points',
                fontsize=8, fontweight='bold', color=INK, va='center',
                ha='right' if right else 'left')

    cb = fig.colorbar(mesh, ax=ax, pad=0.02, fraction=0.04)
    cb.set_label('deviation from unconditional (pp)', color=INK_2, fontsize=8)
    cb.outline.set_visible(False)
    cb.ax.tick_params(color=GRID, labelsize=7.5)

    ax.set_xlabel(f'horizon (+{ws.horizon_unit})')
    ax.set_ylabel('barrier Δ (%)')
    for side in ('top', 'right', 'left', 'bottom'):
        ax.spines[side].set_visible(False)

    condition = cube["meta"]["bin_labels"][b].replace("x", feature_label(head["id"]))
    _title(fig, f'Chance deviates by {abs(cell["dev"]):.1f} pp, when {condition}',
           f'This is the conditional surface minus the baseline. Red means the barrier '
           f'is reached more often than usual; blue means less. The ring is the '
           f'strongest actionable cell.')
    _note(fig, f'{cube["meta"]["bin_labels"][b]} · ring at +{cell["horizon"]}'
               f'{ws.horizon_unit}, q = {head["gate"]["q_value"]:.2g} after '
               f'Benjamini-Hochberg · bin holds {cell["bin_n"]} bars · '
               f'02_shift_array/{head["family"]}/{head["id"]}.npz')
    fig.subplots_adjust(top=0.80)
    return _save(fig, out_path)


def fig_shift(ws, universe, cleared, out: Path) -> Path | None:
    """The strongest cleared node's strongest bin, as a deviation from the baseline."""
    head = _headline_node(ws, cleared, universe)
    if head is None:
        return None
    return _render_shift(ws, head, out / f'C_shift_{head["id"]}.png')


def fig_shift_all(ws, universe, cleared, out: Path) -> list[Path]:
    """Render one conditional-shift figure for every node that cleared both filters."""
    paths: list[Path] = []
    for row in cleared.get('cleared', []):
        if not row.get('best_cell'):
            continue
        try:
            node = find_node(universe, row['node'])
            head = {**node, 'cell': row['best_cell'], 'gate': row}
            suffix = f'{node["id"]}_bin{int(row.get("bin_number", row["best_cell"]["bin"] + 1))}'
            path = _render_shift(ws, head, out / f'C_shift_{suffix}.png')
        except Exception:
            continue
        if path is not None:
            paths.append(path)
    return paths


def fig_atr_ladder(ws, cleared, out: Path) -> Path | None:
    """
    A single actionable cell turned into a decile ladder.

    The strongest ATR cell already appears in the conditional surface. This view removes
    every other Δ/horizon and asks the simpler question: at the same target, how much
    does the event rate move as ATR moves from calm to wide?
    """
    family, node_id = 'volatility', 'atr_14'
    path = ws.shift_cube_path(family, node_id)
    if not path.exists():
        return None

    rows = [r for r in cleared.get('cleared', [])
            if r.get('node') == node_id and r.get('best_cell')]
    if not rows:
        return None
    row = max(rows, key=lambda r: abs(r['best_cell']['dev']))
    cell = row['best_cell']

    cube = shift.load(path)
    th, ts = cube['Δs'], cube['horizons']
    i = int(np.argmin(np.abs(th - float(cell['Δ']))))
    j = int(np.argmin(np.abs(ts - int(cell['horizon']))))

    probs = cube['prob'][i, :, j]
    base = float(cube['base'][i, j])
    labels = cube['meta']['bin_labels']
    x = np.arange(len(probs))
    dev = probs - base

    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    colours = [S1 if d < 0 else S2 for d in dev]
    ax.axhline(base, color=INK_2, linewidth=1.0, zorder=1)
    ax.vlines(x, base, probs, color=FAINT, linewidth=3.0, zorder=2)
    ax.scatter(x, probs, s=70, c=colours, edgecolors=SURFACE, linewidths=1.5, zorder=4)

    lo = int(np.nanargmin(probs))
    hi = int(np.nanargmax(probs))
    endpoints = []
    for k in (lo, hi):
        if k == 0:
            endpoints.append((k, 'left', 8))
        elif k == len(probs) - 1:
            endpoints.append((k, 'left', 8))
        elif k == lo:
            endpoints.append((k, 'right', -8))
        else:
            endpoints.append((k, 'left', 8))
    for k, ha, dx in endpoints:
        ax.annotate(f'{probs[k]:.1%}', (x[k], probs[k]),
                    xytext=(dx, 0), textcoords='offset points',
                    ha=ha, va='center', fontsize=10, fontweight='bold',
                    color=colours[k])
    ax.annotate(f'baseline {base:.1%}', (len(probs) - 1, base),
                xytext=(8, 0), textcoords='offset points',
                ha='left', va='center', fontsize=8.5, fontweight='bold',
                color=INK_2, annotation_clip=False)

    ax.set_xticks(x)
    tick_labels = [f'D{k + 1}' for k in x]
    tick_labels[lo] = f'D{lo + 1}\ncalmest'
    tick_labels[hi] = f'D{hi + 1}\nwidest'
    ax.set_xticklabels(tick_labels)
    ax.set_ylabel(f'P(touch {_Δ_pct(cell["Δ"], ws.delta_step)} within '
                  f'+{cell["horizon"]}{ws.horizon_unit})')
    ax.set_ylim(max(0, float(np.nanmin(probs)) - 0.08),
                min(1, float(np.nanmax(probs)) + 0.10))
    _frame(ax, grid_axis='y')

    _title(fig,
           f'Calm ATR cuts {_Δ_pct(cell["Δ"], ws.delta_step)} touches '
           f'from {base:.0%} to {probs[lo]:.0%}',
           f'At +{cell["horizon"]}{ws.horizon_unit}, {feature_label(node_id)} forms a '
           f'risk ladder: the calmest bin is {probs[lo]:.1%}, the unconditional rate is '
           f'{base:.1%}, and the widest bin is {probs[hi]:.1%}.')
    _note(fig, f'{ws.dir.name} · {labels[lo]} vs {labels[hi]} · '
               f'02_shift_array/{family}/{node_id}.npz')
    fig.subplots_adjust(top=0.78, right=0.84, bottom=0.15)
    return _save(fig, out / 'C_atr_regime_ladder.png')

