"""Single-node report figures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

from engine import barrier
from engine.report_common import full_baseline as _full_baseline
from engine.report_common import headline_node as _headline_node
from engine.report_common import theta_pct as _theta_pct
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
from pipeline.runtime import loader, node_feature


def fig_band(ws, tree, cleared, out: Path) -> Path | None:
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
    head = _headline_node(ws, cleared, tree)
    if head is None:
        return None
    cube = barrier.load_cube(ws.cube_path(head['family'], head['id']))
    base_cube = _full_baseline(ws)

    prob, th, hz = cube['prob'], cube['thetas'], cube['horizons']
    n_bins = prob.shape[1]
    if n_bins < 2:
        return None

    # the live condition: where the feature sits on the last bar it can be computed for
    data, feat = node_feature(ws, head, loader(ws))
    valid = feat.dropna()
    if valid.empty:
        return None
    now_val = float(valid.iloc[-1])
    now_bin = int(np.searchsorted(cube['edges'], now_val)) if len(cube['edges']) else 0
    now_bin = int(np.clip(now_bin, 0, n_bins - 1))

    price = data['close'].reindex(valid.index).dropna()
    n_hist = int(min(len(price), max(40, 6 * int(hz.max()))))
    hist = price.iloc[-n_hist:]
    last = float(hist.iloc[-1])
    step = pd.Series(hist.index).diff().median()
    fwd = [hist.index[-1] + step * int(h) for h in hz]

    # The 50% band is the one level that stays inside the ladder at every horizon and
    # every decile -- 25% and 10% run past ±20% on BTC within a fortnight, and a band
    # clipped at the edge of the measured grid draws a flat ceiling that looks like a
    # finding and is only the ladder ending.
    Q = 0.50
    labels = cube['meta']['bin_labels']
    up_c, dn_c = barrier.touch_band(prob[:, now_bin, :], th, Q)
    up_lo, dn_lo = barrier.touch_band(prob[:, 0, :], th, Q)
    up_hi, dn_hi = barrier.touch_band(prob[:, n_bins - 1, :], th, Q)

    fig, ax = plt.subplots(figsize=(9.6, 5.0))

    # history
    ax.plot(hist.index, hist.to_numpy(), color=INK, linewidth=1.4, zorder=6)

    # The envelope goes in price, not percent, so it reads off the same axis as the line
    # it continues. The filled band is the decile the asset is in *now*; the two thin
    # pairs bracket it with the calmest and widest deciles, because today's regime alone
    # says nothing about how much conditioning is worth -- and today's happens to be a
    # middling decile, where the conditional band and the unconditional one nearly
    # coincide. The bracket is what shows the range the condition spans.
    ax.fill_between(fwd, last * (1 - dn_c), last * (1 + up_c),
                    color=SEQ[1], linewidth=0, zorder=3)
    for arm, colour in ((last * (1 + up_lo), SEQ[9]), (last * (1 - dn_lo), SEQ[9]),
                        (last * (1 + up_hi), SEQ[5]), (last * (1 - dn_hi), SEQ[5])):
        ax.plot(fwd, arm, color=SURFACE, linewidth=3.0, zorder=4)
        ax.plot(fwd, arm, color=colour, linewidth=1.3, zorder=5)

    ax.axvline(hist.index[-1], color=MUTED, linewidth=0.9, zorder=2)
    ax.plot([hist.index[-1]], [last], marker='o', markersize=5.5, color=INK,
            markeredgecolor=SURFACE, markeredgewidth=1.4, zorder=8)

    ax.set_ylabel(f'{ws.asset["ticker"]} close')
    _frame(ax, grid_axis='y')

    j = len(hz) - 1
    _place_labels(ax, fwd[-1], [
        (last * (1 + up_c[j]), f'current band  +{up_c[j]:.0%}   {last * (1 + up_c[j]):,.0f}', SEQ[8]),
        (last * (1 - dn_c[j]), f'current band  −{dn_c[j]:.0%}   {last * (1 - dn_c[j]):,.0f}', SEQ[8]),
        (last * (1 + up_lo[j]), f'calmest decile  +{up_lo[j]:.0%}', SEQ[9]),
        (last * (1 - dn_lo[j]), f'calmest decile  −{dn_lo[j]:.0%}', SEQ[9]),
        (last * (1 + up_hi[j]), f'widest decile  +{up_hi[j]:.0%}', SEQ[5]),
        (last * (1 - dn_hi[j]), f'widest decile  −{dn_hi[j]:.0%}', SEQ[5]),
    ])
    ax.annotate(f'{last:,.0f}', (hist.index[-1], last), xytext=(-8, 11),
                textcoords='offset points', ha='right', fontsize=8.5,
                fontweight='bold', color=INK)

    arms = [up_c, up_lo, up_hi], [dn_c, dn_lo, dn_hi]
    hi_y = max(float(hist.max()), last * (1 + max(np.nanmax(a) for a in arms[0])))
    lo_y = min(float(hist.min()), last * (1 - max(np.nanmax(a) for a in arms[1])))
    pad = (hi_y - lo_y) * 0.07
    ax.set_ylim(lo_y - pad, hi_y + pad)

    _title(fig, 'The measured envelope starts at the last close',
           f'The filled band is the 50% historical touch envelope when '
           f'{feature_label(head["id"])} sits in today\'s bin: {labels[now_bin]}. '
           f'The thin pairs show the calmest and widest bins, so the spread is the '
           f'visible effect of conditioning.')
    _note(fig, f'{ws.asset["ticker"]} through {hist.index[-1].date()} · not a forecast · '
               f'50% touch probability, inverted from 01_surface_array/{head["family"]}/'
               f'{head["id"]}.npz over +1..+{int(hz[j])}{ws.horizon_unit} · current bin '
               f'holds {int(cube["bin_n"][now_bin, j])} bars')
    fig.subplots_adjust(top=0.78, right=0.78, bottom=0.10)
    return _save(fig, out / '01_band.png')


def fig_baseline(ws, out: Path) -> Path | None:
    """
    The reference every other number is read against, as a picture.

    θ = 0 is dropped: it asks whether the high ever returned to the entry close, which is
    true ~99% of the time, and keeping it would spend most of the colour range on a row
    that carries no information.
    """
    cube = _full_baseline(ws)
    surf, th, hz = cube['prob'][:, 0, :], cube['thetas'], cube['horizons']
    keep = np.abs(th) > 1e-12
    surf, th = surf[keep], th[keep]

    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    mesh = ax.pcolormesh(hz, th * 100, surf, cmap=CMAP_SEQ, vmin=0, vmax=1,
                         shading='nearest')
    cs = ax.contour(hz, th * 100, surf, levels=[0.25, 0.5, 0.75],
                    colors=[SURFACE], linewidths=0.9)
    ax.clabel(cs, fmt=lambda v: f'{v:.0%}', fontsize=7, inline=True)
    ax.axhline(0, color=SURFACE, linewidth=1.4)

    cb = fig.colorbar(mesh, ax=ax, pad=0.02, fraction=0.04)
    cb.set_label('P(touch θ within h)', color=INK_2, fontsize=8)
    cb.outline.set_visible(False)
    cb.ax.tick_params(color=GRID, labelsize=7.5)

    ax.set_xlabel(f'horizon (+{ws.horizon_unit})')
    ax.set_ylabel('barrier θ (%)')
    for side in ('top', 'right', 'left', 'bottom'):
        ax.spines[side].set_visible(False)

    up = surf[th > 0]
    dn = surf[th < 0]
    j = len(hz) - 1
    top_theta = th[th > 0][-1]
    bot_theta = th[th < 0][0]
    ax.annotate(f'{up[-1, j]:.1%}', (hz[j], top_theta * 100),
                xytext=(-10, 0), textcoords='offset points', ha='right',
                va='center', fontsize=8, fontweight='bold', color=SURFACE)
    ax.annotate(f'{dn[0, j]:.1%}', (hz[j], bot_theta * 100),
                xytext=(-10, 0), textcoords='offset points', ha='right',
                va='center', fontsize=8, fontweight='bold', color=INK_2)
    _title(fig, 'The baseline already has a direction',
           f'With no condition at all, {_theta_pct(top_theta, ws.theta_step)} by '
           f'+{int(hz[j])}{ws.horizon_unit} is touched {up[-1, j]:.1%} of the time; '
           f'{_theta_pct(bot_theta, ws.theta_step)} is touched {dn[0, j]:.1%}. '
           f'This is the reference every condition must beat.')
    _note(fig, f'{ws.asset["ticker"]} {ws.asset.get("interval", "1d")} · '
               f'01_surface_array/_base/baseline.npz · θ = 0 omitted (near-degenerate by '
               f'construction)')
    fig.subplots_adjust(top=0.80)
    return _save(fig, out / '07_baseline_surface.png')


def fig_shift(ws, tree, cleared, out: Path) -> Path | None:
    """The strongest cleared node's strongest bin, as a deviation from the baseline."""
    head = _headline_node(ws, cleared, tree)
    if head is None:
        return None
    cube = barrier.load_cube(ws.cube_path(head['family'], head['id']))
    base = _full_baseline(ws)['prob'][:, 0, :]
    b = int(head['cell']['bin'])

    th, hz = cube['thetas'], cube['horizons']
    keep = np.abs(th) > 1e-12
    dev = (cube['prob'][:, b, :] - base)[keep] * 100.0
    th = th[keep]
    lim = float(np.nanmax(np.abs(dev)))

    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    mesh = ax.pcolormesh(hz, th * 100, dev, cmap=CMAP_DIV,
                         norm=TwoSlopeNorm(vcenter=0.0, vmin=-lim, vmax=lim),
                         shading='nearest')
    ax.axhline(0, color=SURFACE, linewidth=1.4)

    cell = head['cell']
    ax.plot([cell['horizon']], [cell['theta'] * 100], marker='o', markersize=7,
            markerfacecolor='none', markeredgecolor=INK, markeredgewidth=1.4)
    # a cell near the right edge would push its label under the colorbar
    right = cell['horizon'] > hz.min() + 0.7 * (hz.max() - hz.min())
    ax.annotate(f"{cell['dev']:+.1f} pp", (cell['horizon'], cell['theta'] * 100),
                xytext=(-10 if right else 10, 0), textcoords='offset points',
                fontsize=8, fontweight='bold', color=INK, va='center',
                ha='right' if right else 'left')

    cb = fig.colorbar(mesh, ax=ax, pad=0.02, fraction=0.04)
    cb.set_label('deviation from unconditional (pp)', color=INK_2, fontsize=8)
    cb.outline.set_visible(False)
    cb.ax.tick_params(color=GRID, labelsize=7.5)

    ax.set_xlabel(f'horizon (+{ws.horizon_unit})')
    ax.set_ylabel('barrier θ (%)')
    for side in ('top', 'right', 'left', 'bottom'):
        ax.spines[side].set_visible(False)

    direction = 'raises' if cell['dev'] > 0 else 'cuts'
    target = _theta_pct(cell['theta'], ws.theta_step)
    _title(fig, f'{feature_label(head["id"])} {direction} {target} touches by {abs(cell["dev"]):.1f} pp',
           f'This is the conditional surface minus the baseline. Red means the barrier '
           f'is reached more often than usual; blue means less. The ring is the '
           f'strongest actionable cell.')
    _note(fig, f'{cube["meta"]["bin_labels"][b]} · ring at +{cell["horizon"]}'
               f'{ws.horizon_unit}, q = {head["gate"]["q_value"]:.2g} after '
               f'Benjamini-Hochberg · bin holds {cell["bin_n"]} bars · '
               f'01_surface_array/{head["family"]}/{head["id"]}.npz')
    fig.subplots_adjust(top=0.80)
    return _save(fig, out / '08_conditional_shift.png')


def fig_atr_ladder(ws, cleared, out: Path) -> Path | None:
    """
    A single actionable cell turned into a decile ladder.

    The strongest ATR cell already appears in the conditional surface. This view removes
    every other theta/horizon and asks the simpler question: at the same target, how much
    does the event rate move as ATR moves from calm to wide?
    """
    family, node_id = 'volatility', 'atr_14'
    path = ws.cube_path(family, node_id)
    if not path.exists():
        return None

    rows = [r for r in cleared.get('cleared', [])
            if r.get('node') == node_id and r.get('best_cell')]
    if not rows:
        return None
    row = max(rows, key=lambda r: abs(r['best_cell']['dev']))
    cell = row['best_cell']

    cube = barrier.load_cube(path)
    base_cube = _full_baseline(ws)
    th, hz = cube['thetas'], cube['horizons']
    i = int(np.argmin(np.abs(th - float(cell['theta']))))
    j = int(np.argmin(np.abs(hz - int(cell['horizon']))))

    probs = cube['prob'][i, :, j]
    base = float(base_cube['prob'][i, 0, j])
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
    ax.set_ylabel(f'P(touch {_theta_pct(cell["theta"], ws.theta_step)} within '
                  f'+{cell["horizon"]}{ws.horizon_unit})')
    ax.set_ylim(max(0, float(np.nanmin(probs)) - 0.08),
                min(1, float(np.nanmax(probs)) + 0.10))
    _frame(ax, grid_axis='y')

    _title(fig,
           f'Calm ATR cuts {_theta_pct(cell["theta"], ws.theta_step)} touches '
           f'from {base:.0%} to {probs[lo]:.0%}',
           f'At +{cell["horizon"]}{ws.horizon_unit}, {feature_label(node_id)} forms a '
           f'risk ladder: the calmest bin is {probs[lo]:.1%}, the unconditional rate is '
           f'{base:.1%}, and the widest bin is {probs[hi]:.1%}.')
    _note(fig, f'{ws.dir.name} · {labels[lo]} vs {labels[hi]} · '
               f'01_surface_array/{family}/{node_id}.npz + 01_surface_array/_base/baseline.npz')
    fig.subplots_adjust(top=0.78, right=0.84, bottom=0.15)
    return _save(fig, out / '11_atr_regime_ladder.png')

