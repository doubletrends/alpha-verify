"""
Stage nine: the figures, rendered from the artifacts on disk.

Nothing here measures anything. Every number in every figure is read back out of a
`.npz` or a `.json` that an earlier stage wrote, which is the same discipline the
workbook renderer follows and for the same reason: a picture that recomputes its own
inputs can disagree with the file it claims to illustrate.

The one exception is the null histogram, which needs the whole shift distribution rather
than the three numbers the validation artifact stores. It re-derives that distribution
from the node's own cube coordinates and edges, so it is the same test on the same grid,
just not reduced.

The visual grammar is fixed across every figure so the set reads as one system:

  magnitude   one hue, light to dark          (probabilities, counts)
  polarity    blue/red with a neutral middle  (deviations, skill against a reference)
  identity    fixed categorical slots         (the three composition models)

Colours are the validated default palette; the categorical slots are used in order and
never cycled. Everything renders on a light surface at 2x, because these are read in a
README on a phone as often as on a monitor.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from engine import barrier, bayes, validate as val
from engine.writer import feature_label
from tree.tree import load_tree, all_nodes, find_node
from workspace import BASELINE_NODE

# ── the visual system ─────────────────────────────────────────────────────────

SURFACE = '#fcfcfb'
INK     = '#0b0b0b'
INK_2   = '#52514e'
MUTED   = '#8a8983'
GRID    = '#e6e5e1'
FAINT   = '#d8d7d2'

# categorical slots, in fixed order, never cycled
S1, S2, S3 = '#2a78d6', '#eb6834', '#1baf7a'

# one hue, light to dark: the sequential ramp for magnitude
SEQ = ['#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7',
       '#3987e5', '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b']
# ordinal steps start at #86b6ef -- no lighter, or the first stage recedes into the page
ORD = ['#86b6ef', '#5598e7', '#2a78d6', '#1c5cab', '#104281']

CMAP_SEQ = LinearSegmentedColormap.from_list('seq', SEQ)
# two hues that read as opposite, with a neutral grey middle: never a hue at the midpoint
CMAP_DIV = LinearSegmentedColormap.from_list('div', [
    '#0d366b', '#1c5cab', '#2a78d6', '#86b6ef', '#cde2fb',
    '#f0efec',
    '#fbd9d8', '#f5aeac', '#ec7d7b', '#e34948', '#b62f2e'])

plt.rcParams.update({
    'figure.dpi': 200,
    'savefig.dpi': 200,
    'figure.facecolor': SURFACE,
    'axes.facecolor': SURFACE,
    'savefig.facecolor': SURFACE,
    'font.family': 'DejaVu Sans',
    'font.size': 8.5,
    'text.color': INK,
    'axes.labelcolor': INK_2,
    'axes.edgecolor': GRID,
    'axes.linewidth': 0.8,
    'xtick.color': INK_2,
    'ytick.color': INK_2,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.frameon': False,
    'axes.grid': False,
})


def _frame(ax, grid_axis: str | None = 'y') -> None:
    """Hairline, recessive chrome: two spines, one solid grid, nothing dashed."""
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(GRID)
    if grid_axis:
        ax.set_axisbelow(True)
        ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.6, linestyle='-')


def _wrap(text: str, width: int) -> str:
    """
    Hard-wrap each already-broken line to `width` characters.

    savefig(bbox_inches='tight') grows the canvas to contain every artist, so one long
    unwrapped caption silently stretches the figure to twice its intended aspect. Wrapping
    here rather than by hand in each caption keeps the whole set to one shape.
    """
    return '\n'.join(textwrap.fill(line, width) if line else ''
                     for line in text.split('\n'))


def _title(fig, title: str, subtitle: str, y: float = 0.98) -> None:
    fig.text(0.012, y, _wrap(title, 82), ha='left', va='top', fontsize=12.5,
             fontweight='bold', color=INK)
    fig.text(0.012, y - 0.055, _wrap(subtitle, 108), ha='left', va='top',
             fontsize=8.5, color=INK_2)


def _note(fig, text: str) -> None:
    # Below the axes rather than inside the bottom margin: bbox_inches='tight' grows the
    # canvas to include it either way, and at y > 0 it lands on the x-axis label. The
    # offset is a fixed physical distance rather than a figure fraction, or a tall figure
    # opens a gap several times the gap on a short one.
    fig.text(0.012, -0.3 / fig.get_figheight(), _wrap(text, 132), ha='left', va='top',
             fontsize=7.2, color=MUTED)


def _place_labels(ax, x, items: list, min_gap_frac: float = 0.055) -> None:
    """
    Right-edge direct labels, nudged apart so none sits on top of another.

    Series that converge -- and two of these are a conditional band against its own
    unconditional reference, which often nearly touch -- put their end labels within a
    few pixels of each other. One pass from the bottom up, pushing any label that lands
    too close to its predecessor, is enough here and keeps every label attached to the
    value it names.
    """
    lo, hi = ax.get_ylim()
    gap = (hi - lo) * min_gap_frac
    items = sorted(items, key=lambda it: it[0])
    placed = []
    for y, text, colour in items:
        if placed and y - placed[-1] < gap:
            y = placed[-1] + gap
        placed.append(y)
        ax.annotate(text, (x, y), xytext=(7, 0), textcoords='offset points',
                    color=colour, fontsize=8, fontweight='bold', va='center',
                    annotation_clip=False)


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches='tight', pad_inches=0.28)
    plt.close(fig)
    return path


# ── shared reads ──────────────────────────────────────────────────────────────

def _pct(v: float) -> str:
    return f'{v:+.0%}'


def _theta_pct(v: float, step: float) -> str:
    """Barrier label: keep half-percent grids distinct, but avoid noisy .0% labels."""
    decimals = 1 if step < 0.01 else 0
    return format(float(v), f'+.{decimals}%')


def _full_baseline(ws) -> dict:
    """
    The baseline node's *full* cube, not its summary.

    The figures render against the full grid because that is the faithful record and
    because the full ladder draws a smooth cone where the summary ladder draws a
    staircase. The summary grid exists to be judged on, and nothing here judges.
    """
    return barrier.load_cube(ws.cube_path('_base', BASELINE_NODE))


def _headline_node(ws, cleared: dict, tree: dict) -> dict | None:
    """
    The node the single-node figures are drawn for: the cleared node whose best cell
    moves the barrier rate furthest.

    Chosen from the gate's own output rather than by hand, so the illustration is
    whatever the pipeline actually ranked first and cannot drift away from it.
    """
    rows = [c for c in cleared.get('cleared', []) if c.get('best_cell')]
    if not rows:
        return None
    best = max(rows, key=lambda c: abs(c['best_cell']['dev']))
    node = find_node(tree, best['node'])
    return {**node, 'cell': best['best_cell'], 'gate': best}


# ── 1. the band ───────────────────────────────────────────────────────────────

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
    import run as _run

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
    data, feat = _run._node_feature(ws, head, _run._loader(ws))
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


# ── 2. the landscape ──────────────────────────────────────────────────────────

def fig_landscape(ws, tree, evaluation, cleared, out: Path) -> Path | None:
    """
    Every node as one point, on the three axes the pipeline actually judges by.

    x and y are each a node's peak against *its own* null, so both are dimensionless and
    1.0 means "exactly at the 95th percentile of what shuffling produces". That framing
    matters more than it looks: the noise ceiling is not one number across the sweep, it
    depends on how a feature bins and how much history it survives, so comparing raw
    percentage points between nodes compares their sample sizes as much as their signal.

    x  does the condition move the *magnitude* of the move -- P(touch θ) at all
    y  does it move the *direction* -- the up/down asymmetry, drift removed
    c  is it big enough to act on -- the economic filter's own effect size

    The picture is the thesis. Everything spreads horizontally and nothing spreads
    vertically: conditions move how far price travels and not which way.
    """
    tests = cleared.get('tests', [])
    if not tests:
        return None

    def best_ratio(rows):
        out_ = {}
        for r in rows:
            if r.get('peak_p95'):
                v = r['peak_real'] / r['peak_p95']
                out_[r['node']] = max(out_.get(r['node'], -np.inf), v)
        return out_

    touch = best_ratio(tests)
    skew = best_ratio(cleared.get('skew_tests', []))
    ev = evaluation.get('nodes', {})
    fam_of = {n['id']: n['family'] for n in all_nodes(tree)}

    pts = []
    for node, x in touch.items():
        y = skew.get(node)
        if y is None or node == BASELINE_NODE:
            continue
        best = ev.get(node, {}).get('best')
        pts.append((node, x, y, abs(best['dev']) if best else 0.0, fam_of.get(node, '?')))
    if len(pts) < 5:
        return None

    xs = np.array([p[1] for p in pts])
    ys = np.array([p[2] for p in pts])
    cs = np.array([p[3] for p in pts])

    # Equal aspect fixes the box's shape to the data's, so the *figure* has to follow or
    # a tall cloud gets a square box stranded in a wide canvas. BTC's cloud is flat and
    # wide; the hourly index has a genuine vertical outlier and comes out nearly square.
    span_x = (xs.max() - xs.min()) * 1.14
    span_y = (ys.max() - ys.min()) * 1.28
    fig_h = float(np.clip(7.0 * span_y / span_x + 2.0, 4.2, 8.4))
    fig, ax = plt.subplots(figsize=(9.4, fig_h))
    ax.axvline(1.0, color=GRID, linewidth=1.0, zorder=1)
    ax.axhline(1.0, color=GRID, linewidth=1.0, zorder=1)
    sc = ax.scatter(xs, ys, c=cs, cmap=CMAP_SEQ, vmin=0, vmax=max(cs.max(), 1),
                    s=64, linewidths=1.4, edgecolors=SURFACE, zorder=5)

    cb = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.04)
    cb.set_label('strongest cell vs the unconditional rate (pp)', color=INK_2, fontsize=8)
    cb.outline.set_visible(False)
    cb.ax.tick_params(color=GRID, labelsize=7.5)

    ax.set_xlabel('moves the barrier rate   →   peak ÷ its own null p95')
    ax.set_ylabel('moves direction   →\npeak ÷ its own null p95')
    _frame(ax, grid_axis='both')

    # Both axes are the same dimensionless quantity -- a peak in units of that node's own
    # noise ceiling -- so a unit has to be the same length on each. Letting them autoscale
    # independently stretches a vertical range of 0.9 to the same height as a horizontal
    # range of 2.6 and draws a round cloud, which is the opposite of what the data says.
    # Equal aspect with each axis held to its own data range gives a wide, short box: no
    # empty quadrant, and the flattening is the finding rather than a drawing choice.
    px = (xs.max() - xs.min()) * 0.07
    py = (ys.max() - ys.min()) * 0.14
    ax.set_xlim(xs.min() - px, xs.max() + px)
    ax.set_ylim(ys.min() - py, ys.max() + py)
    ax.set_aspect('equal', adjustable='box')
    ax.annotate('noise ceiling, both axes', (1.0, 1.0), xytext=(5, 3),
                textcoords='offset points', color=MUTED, fontsize=7.5,
                va='bottom', ha='left')

    # One in-plot callout only, and it is the one that has to point at something. The
    # rest of the naming goes in the note: the cloud changes shape per workspace, so any
    # fixed in-plot corner is occupied on one of them.
    dead = sorted({f for f in {p[4] for p in pts}
                   if not any(t.get('verdict') == 'discovery'
                              for t in tests if fam_of.get(t['node']) == f)})
    if dead:
        din = [i for i, p in enumerate(pts) if p[4] in dead]
        ax.annotate(' · '.join(dead) + '\ncleared nothing',
                    (float(xs[din].mean()), float(ys[din].min())),
                    xytext=(-16, -24), textcoords='offset points', ha='right',
                    fontsize=7.8, fontweight='bold', color=INK, linespacing=1.5,
                    arrowprops=dict(arrowstyle='-', color=MUTED, linewidth=0.8))

    top_x = int(np.argmax(xs))
    ax.annotate(f'strongest magnitude\n{pts[top_x][0]}',
                (float(xs[top_x]), float(ys[top_x])),
                xytext=(-8, 18), textcoords='offset points', ha='right',
                fontsize=7.8, fontweight='bold', color=INK,
                arrowprops=dict(arrowstyle='-', color=MUTED, linewidth=0.8))

    # Whether anything moves direction is the whole question, and the answer differs by
    # workspace: BTC's cloud never leaves the line, the hourly index has VIX nodes well
    # clear of it. A fixed headline would be false on one of the two.
    n_dir = int((ys > 1.0).sum())
    exp = len(pts) * (1 - 0.95 ** len(ws.summary_horizons))
    lifted = [pts[k][0] for k in np.argsort(-ys) if ys[k] > 1.5]
    if lifted:
        head_txt = 'Magnitude clears the null; direction has exceptions'
        vert = (f'vertically all but {len(lifted)} sit on it, and those {len(lifted)} '
                f'({", ".join(lifted[:3])}) are what the gate picks up')
    else:
        head_txt = 'Magnitude clears the null; direction mostly does not'
        vert = (f'vertically it never leaves the line — {n_dir} of {len(pts)} nodes '
                f'cross it at all, against {exp:.0f} expected by chance across '
                f'{len(ws.summary_horizons)} horizons')
    _title(fig, head_txt,
           f'One point per node. Both axes are peak ÷ that node\'s shuffled-null p95, '
           f'so 1.0 is the noise ceiling on each. The horizontal spread reaches '
           f'{xs.max():.1f}x; {vert}.')
    _note(fig, f'{ws.dir.name} · furthest on magnitude: '
               f'{" · ".join(pts[k][0] for k in np.argsort(-xs)[:3])} · each axis takes '
               f'the node\'s strongest horizon · crossing 1.0 once across '
               f'{len(ws.summary_horizons)} horizons is weak; distance past it is the '
               f'claim · 05_gate.json + evaluation.json')
    fig.subplots_adjust(top=1 - 1.05 / fig_h, bottom=0.72 / fig_h)
    return _save(fig, out / '02_landscape.png')


# ── 7. the unconditional surface ──────────────────────────────────────────────

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


# ── 8. what a condition does to it ────────────────────────────────────────────

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


# ── 11. one condition as an economic ladder ───────────────────────────────────

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


def fig_null_gap_ranking(ws, cleared, out: Path) -> Path | None:
    """
    The figure-5 argument across the strongest node/horizon claims.

    One histogram can look hand-picked. Ranking each node by its largest discovered gap
    against its own null keeps the same visual argument -- observed far beyond null p95 --
    while showing that the top of the sweep has depth.
    """
    tests = cleared.get('tests', [])
    if not tests:
        return None

    best = {}
    for r in tests:
        if r.get('verdict') != 'discovery' or not r.get('peak_p95') or r['node'] == BASELINE_NODE:
            continue
        ratio = float(r['peak_real']) / float(r['peak_p95'])
        cur = best.get(r['node'])
        if cur is None or ratio > cur[0]:
            best[r['node']] = (ratio, r)
    rows = sorted(best.values(), key=lambda x: x[0], reverse=True)[:12]
    if len(rows) < 3:
        return None

    ratios = np.array([r[0] for r in rows])
    tests_top = [r[1] for r in rows]
    observed = np.array([float(r['peak_real']) for r in tests_top])
    p95 = np.array([float(r['peak_p95']) for r in tests_top])
    y = np.arange(len(rows))[::-1]

    fig_h = 0.36 * len(rows) + 2.1
    fig, ax = plt.subplots(figsize=(8.4, fig_h))
    ax.hlines(y, p95, observed, color=SEQ[3], linewidth=3.0, zorder=2)
    ax.plot(p95, y, linestyle='None', marker='|', markersize=14,
            markeredgewidth=1.7, color=INK_2, zorder=3, label='null p95')
    ax.scatter(observed, y, s=62, color=S1, edgecolors=SURFACE,
               linewidths=1.4, zorder=4, label='observed')

    for yy, obs, ratio in zip(y, observed, ratios):
        ax.annotate(f'{ratio:.1f}x', (obs, yy), xytext=(8, 0),
                    textcoords='offset points', ha='left', va='center',
                    fontsize=8.2, fontweight='bold', color=S1)

    labels = [f'{feature_label(r["node"])}  +{r["horizon"]}{ws.horizon_unit}'
              for r in tests_top]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.2)
    ax.set_xlabel('peak |deviation| over the surface (pp)')
    ax.set_xlim(0, float(np.nanmax(observed)) * 1.25)
    _frame(ax, grid_axis='x')
    ax.legend(loc='lower right', fontsize=8, labelcolor=INK_2)

    at_floor = sum(1 for r in tests_top if r.get('at_floor'))
    _title(fig,
           f'Top effects sit {ratios[-1]:.1f}–{ratios[0]:.1f}x beyond their own null',
           f'Each row keeps one node: its strongest discovered horizon. Grey ticks are '
           f'the shuffled-null p95; blue dots are the observed peak. {at_floor} of these '
           f'{len(rows)} are already at the exact p-value floor.')
    _note(fig, f'{ws.dir.name} · ranked from 05_gate.json tests · one best discovered '
               f'horizon per node, so repeated variants do not fill the chart')
    fig.subplots_adjust(top=1 - 0.95 / fig_h, left=0.25, bottom=0.14)
    return _save(fig, out / '12_null_gap_ranking.png')


# ── 6. the funnel ─────────────────────────────────────────────────────────────

def fig_funnel(ws, tree, evaluation, cleared, out: Path) -> Path:
    n_nodes = len(all_nodes(tree))
    n_econ = len(evaluation.get('passed', []))
    n_disc = len({t['node'] for t in cleared.get('tests', [])
                  if t.get('verdict') == 'discovery'})
    n_clear = len(cleared.get('cleared', []))

    stages = [('measured', n_nodes, 'every node in the universe'),
              ('economically usable', n_econ, f'≥ {ws.min_dev:.0f} pp across ≥ '
                                              f'{ws.min_run} adjacent θ rows'),
              ('survives the null', n_disc, 'BH discovery on the shuffled null'),
              ('both', n_clear, 'the only claims that leave the pipeline')]

    fig, ax = plt.subplots(figsize=(8.0, 2.9))
    ys = np.arange(len(stages))[::-1]
    # the descriptions share one x, past the longest bar, so they read as a column
    # rather than a ragged edge that collides with whichever count precedes it
    desc_x = n_nodes * 1.14
    for (name, v, sub), y, colour in zip(stages, ys, ORD):
        ax.barh(y, v, height=0.62, color=colour, linewidth=0)
        ax.text(v + n_nodes * 0.02, y, f'{v}', va='center', ha='left',
                fontsize=11, fontweight='bold', color=INK)
        ax.text(desc_x, y, sub, va='center', ha='left', fontsize=7.5, color=MUTED)

    ax.set_yticks(ys)
    ax.set_yticklabels([s[0] for s in stages], fontsize=9, color=INK)
    ax.set_xlim(0, n_nodes * 1.95)
    ax.set_xticks([])
    for side in ('top', 'right', 'bottom', 'left'):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)

    _title(fig, 'Two filters, both required',
           f'{ws.dir.name} — a node has to be big enough to act on and survive a '
           f'shuffled null corrected across the whole sweep.')
    _note(fig, f'{len(cleared.get("tests", []))} tests over {n_nodes} nodes × '
               f'{len(ws.summary_horizons)} horizons · Benjamini-Hochberg at '
               f'q = {cleared.get("method", {}).get("q", 0.05)} · evaluation.json + '
               f'05_gate.json')
    fig.subplots_adjust(top=0.74, left=0.19)
    return _save(fig, out / '06_funnel.png')


# ── 3. where the edge actually is ─────────────────────────────────────────────

def fig_horizons(ws, cleared, out: Path) -> Path | None:
    rows = cleared.get('cleared', [])
    if not rows:
        return None
    hz = sorted({int(c['horizon']) for c in rows})
    counts = [sum(1 for c in rows if int(c['horizon']) == h) for h in hz]
    top = int(np.argmax(counts))

    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    colours = [S1 if i == top else FAINT for i in range(len(hz))]
    ax.bar(range(len(hz)), counts, width=0.62, color=colours, linewidth=0)
    for i, c in enumerate(counts):
        ax.text(i, c + max(counts) * 0.03, str(c), ha='center', va='bottom',
                fontsize=9, fontweight='bold',
                color=INK if i == top else INK_2)

    ax.set_xticks(range(len(hz)))
    ax.set_xticklabels([f'+{h}{ws.horizon_unit}' for h in hz])
    ax.set_ylabel('cleared nodes')
    ax.set_ylim(0, max(counts) * 1.2)
    _frame(ax, grid_axis='y')

    share = counts[top] / sum(counts)
    _title(fig, 'Read the horizon before believing the headline',
           f'Where each cleared node has its strongest cell. {counts[top]} of '
           f'{sum(counts)} sit at +{hz[top]}{ws.horizon_unit} — {share:.0%} of the '
           f'result.\nOne bar ahead, "will price touch −2%" is close to asking "is '
           f'volatility high right now", and volatility features answer that almost '
           f'tautologically.')
    _note(fig, 'It is real, it is significant, and it is mostly mechanical · '
               '05_gate.json')
    fig.subplots_adjust(top=0.70)
    return _save(fig, out / '03_where_the_edge_is.png')


# ── 5. the null, unreduced ───────────────────────────────────────────────────

def fig_null(ws, tree, cleared, out: Path) -> Path | None:
    """
    The exact null as a distribution rather than a p-value.

    Every shift is a world in which the feature carries nothing but keeps its own
    autocorrelation. The observed surface either sits inside that cloud or it does not,
    and the width of the cloud is the honest answer to "how big is big".
    """
    import run as _run

    head = _headline_node(ws, cleared, tree)
    if head is None:
        return None
    cube = barrier.load_cube(ws.summary_cube_path(head['family'], head['id']))
    h = int(head['gate']['horizon'])

    get_data = _run._loader(ws)
    data, feat = _run._node_feature(ws, head, get_data)
    d = val.peak_shift_distribution(data, feat, h, cube['thetas'], cube['edges'])

    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    ax.hist(d['null'], bins=48, color=FAINT, linewidth=0)
    ax.axvline(d['p95'], color=INK_2, linewidth=1.2)
    ax.axvline(d['observed'], color=S1, linewidth=2.2)

    ymax = ax.get_ylim()[1]
    ax.annotate(f'observed  {d["observed"]:.1f} pp', (d['observed'], ymax * 0.92),
                xytext=(-8, 0), textcoords='offset points', color=S1,
                fontsize=9, fontweight='bold', ha='right', va='top')
    ax.annotate(f'null p95  {d["p95"]:.1f} pp', (d['p95'], ymax * 0.60),
                xytext=(8, 0), textcoords='offset points', color=INK_2,
                fontsize=8, ha='left', va='top')

    ax.set_xlabel('peak |deviation| over the surface (pp)')
    ax.set_ylabel('circular shifts')
    _frame(ax, grid_axis='y')

    at_floor = d['p_value'] <= d['floor'] + 1e-12
    _title(fig, 'The null is exact, and its floor is real',
           f'{feature_label(head["id"])} at +{h}{ws.horizon_unit}. Every one of '
           f'{d["n_shifts"]:,} usable circular shifts, from a single FFT.\n'
           f'p = {d["p_value"]:.2g}'
           + (f' — the floor 1/(n+1), the strongest this test can physically report.'
              if at_floor else '.'))
    _note(fig, f'A shift keeps the feature\'s autocorrelation and destroys only its '
               f'alignment with the future · there are exactly {d["n_shifts"]:,} of them, '
               f'so no p-value below {d["floor"]:.2g} exists · this is why the gate '
               f'controls FDR and not family-wise error')
    fig.subplots_adjust(top=0.78)
    return _save(fig, out / '05_null.png')


# ── 4. which families carry anything ──────────────────────────────────────────

def fig_families(ws, tree, cleared, out: Path) -> Path | None:
    tests = cleared.get('tests', [])
    if not tests:
        return None
    fam_of = {n['id']: n['family'] for n in all_nodes(tree)}
    # the baseline family is excluded: shuffling a constant cannot move anything, so its
    # row is empty by construction and would read as a finding beside the ones that are
    fams = sorted({n['family'] for n in all_nodes(tree)} - {'_base'})
    hz = sorted({int(t['horizon']) for t in tests})

    m = np.zeros((len(fams), len(hz)))
    for t in tests:
        fam = fam_of.get(t['node'])
        if t.get('verdict') != 'discovery' or fam not in fams:
            continue
        m[fams.index(fam), hz.index(int(t['horizon']))] += 1

    order = np.argsort(-m.sum(axis=1), kind='stable')
    m, fams = m[order], [fams[i] for i in order]

    # a zero is an absence, not the lightest step of a magnitude ramp: leave it blank
    shown = np.where(m > 0, m, np.nan)
    cmap = CMAP_SEQ.copy()
    cmap.set_bad(SURFACE)

    fig, ax = plt.subplots(figsize=(7.6, 0.32 * len(fams) + 2.0))
    ax.pcolormesh(np.arange(len(hz) + 1), np.arange(len(fams) + 1), shown,
                  cmap=cmap, vmin=0, vmax=max(m.max(), 1))
    for i in range(len(fams)):
        for j in range(len(hz)):
            if m[i, j] > 0:
                ax.text(j + 0.5, i + 0.5, f'{int(m[i, j])}', ha='center', va='center',
                        fontsize=7.5, fontweight='bold',
                        color=SURFACE if m[i, j] > m.max() * 0.55 else INK)

    ax.set_xticks(np.arange(len(hz)) + 0.5)
    ax.set_xticklabels([f'+{h}{ws.horizon_unit}' for h in hz])
    ax.set_yticks(np.arange(len(fams)) + 0.5)
    ax.set_yticklabels(fams, fontsize=8)
    ax.invert_yaxis()
    ax.tick_params(length=0)
    for side in ('top', 'right', 'left', 'bottom'):
        ax.spines[side].set_visible(False)

    dead = [f for f, row in zip(fams, m) if row.sum() == 0]
    _title(fig, 'Which families carry anything',
           f'Nodes with a Benjamini-Hochberg discovery, by family and horizon. '
           f'A blank row cleared nothing, anywhere, at any barrier.'
           + (f'\nBlank here: {", ".join(dead)} — '
              f'{sum(1 for n in all_nodes(tree) if n["family"] in dead)} nodes between '
              f'them, and not one significant cell.' if dead else ''))
    _note(fig, f'{ws.dir.name} · 05_gate.json')
    fig.subplots_adjust(top=0.86 - 0.5 / len(fams), left=0.20, bottom=0.10)
    return _save(fig, out / '04_families.png')


# ── 9. composition, out of sample ─────────────────────────────────────────────

def fig_calibration(ws, out: Path) -> Path | None:
    if not ws.bayes_path.exists():
        return None
    z = np.load(ws.bayes_path, allow_pickle=False)
    meta = json.loads(str(z['meta']))
    summary = ws.read_json(ws.bayes_summary_path)
    m = summary.get('metrics', {})

    y = z['y']
    n_bins = int(np.clip(len(y) // 30, 4, 10))
    series = [(f'naive Bayes, all {summary.get("design", {}).get("n_features", "")} nodes',
               'p_all', 'all_nodes', S2),
              ('one node per family', 'p_dedup', 'one_per_family', S3),
              ('  + scale corrected', 'p_scaled', 'one_per_family_scaled', S1)]

    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1.0, zorder=1)
    ax.annotate('perfectly calibrated', (0.62, 0.62), xytext=(4, -12),
                textcoords='offset points', color=MUTED, fontsize=7.5, rotation=38)

    # The three curves converge and their endpoints land within a few percent of each
    # other, so endpoint labels overlap whatever the offset. The scores go in the legend
    # instead, where they sit beside the colour that carries the identity anyway.
    hi = 0.0
    for label, key, mkey, colour in series:
        c = bayes.calibration(y, z[key], n_bins)
        if not len(c['predicted']):
            continue
        ax.plot(c['predicted'], c['realized'], color=colour, linewidth=1.8,
                marker='o', markersize=6.5, markerfacecolor=colour,
                markeredgecolor=SURFACE, markeredgewidth=1.6, zorder=4,
                label=f"{label.strip()}  —  Brier {m[mkey]['brier']:.3f}")
        hi = max(hi, c['predicted'].max(), c['realized'].max())

    prior = m.get('prior_only', {}).get('brier')
    ax.axhline(m.get('realized_rate', np.nan), color=GRID, linewidth=0.9, zorder=0,
               label=f'constant prior  —  Brier {prior:.3f}')

    lim = min(1.0, hi * 1.15)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel('predicted probability (out of sample)')
    ax.set_ylabel('realized frequency')
    ax.legend(loc='upper left', fontsize=8, labelcolor=INK_2)
    _frame(ax, grid_axis='both')

    t = summary.get('target', {})
    d = summary.get('design', {})
    # which models actually beat the constant prior is a fact about this workspace, not a
    # fixed storyline: on BTC the corrected model clears it, on the hourly index it lands
    # just short. Stating it from the numbers keeps the caption true in both.
    beats = [name for name, _, mkey, _ in series if m[mkey]['brier'] < prior]
    if not beats:
        verdict = ('no model beats it — the ranking is real (see the AUC grid) and the '
                   'probabilities still are not')
    elif len(beats) == 1:
        verdict = f'only "{beats[0].strip()}" beats it'
    else:
        verdict = f'{len(beats)} of the three beat it'
    _title(fig, 'Composing the conditions — and the price of assuming they are independent',
           f'P(touch {_theta_pct(t.get("theta", 0), ws.theta_step)} within {t.get("horizon")}'
           f'{t.get("unit", "d")}), {d.get("folds")}-fold expanding walk forward. '
           f'A point above the line was under-predicted, below it over-predicted. '
           f'A constant prior scores Brier {prior:.3f}; {verdict}.')
    _note(fig, f'{m.get("n_scored", 0)} non-overlapping out-of-sample bars · every table, '
               f'edge, prior and scale fit on the training window only, with a '
               f'{d.get("embargo_bars")}-bar embargo · overconfidence factor '
               f'{1 / summary.get("platt_a_mean", 1):.1f}x · 06_bayes.json')
    fig.subplots_adjust(top=0.80)
    return _save(fig, out / '09_composition.png')


def fig_composition_grid(ws, out: Path) -> Path | None:
    """
    Where the composed model *ranks*, over the grid everything else is judged on.

    This plots AUC rather than Brier skill, and the reason is worth stating: Brier skill
    against the prior is unbounded below, and at the corners of the ladder -- a ±20%
    barrier one bar out, which almost never happens -- a handful of confident misses
    drive it to −7 while the best cell on the grid reaches +0.13. On one colour scale
    that is a flat wash with two black corners. AUC is bounded in [0, 1], neutral at
    0.5, and answers the question this figure is actually for: does the ordering survive
    out of sample? Whether the *probabilities* do is fig. 8's job, and the two are
    separate claims.
    """
    if not ws.bayes_path.exists():
        return None
    z = np.load(ws.bayes_path, allow_pickle=False)
    th_all, hz_all, auc = z['grid_theta'], z['grid_horizon'], z['grid_auc']
    if not len(th_all):
        return None

    # A target needs events on both sides before its AUC means anything. At the corners
    # of the ladder -- a +20% barrier one bar out -- the walk forward scores a handful of
    # positives, and the resulting AUC swings to 0.1 or 0.9 on two or three cases. Left
    # in, one such cell sets the colour range and flattens the other hundred.
    MIN_EVENTS = 10
    n_pos = z['grid_realized'] * z['grid_n_scored']
    n_neg = z['grid_n_scored'] - n_pos
    usable = (n_pos >= MIN_EVENTS) & (n_neg >= MIN_EVENTS)

    th = np.array(sorted(set(th_all)))
    hz = np.array(sorted(set(hz_all)))
    m = np.full((len(th), len(hz)), np.nan)
    for t, h, a, ok in zip(th_all, hz_all, auc, usable):
        if ok:
            m[int(np.flatnonzero(th == t)[0]), int(np.flatnonzero(hz == h)[0])] = a
    if not np.isfinite(m).any():
        return None
    lim = float(np.nanmax(np.abs(m - 0.5)))

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    cmap = CMAP_DIV.copy()
    cmap.set_bad(SURFACE)
    mesh = ax.pcolormesh(np.arange(len(hz) + 1), np.arange(len(th) + 1), m,
                         cmap=cmap,
                         norm=TwoSlopeNorm(vcenter=0.5, vmin=0.5 - lim, vmax=0.5 + lim))
    cb = fig.colorbar(mesh, ax=ax, pad=0.02, fraction=0.045)
    cb.set_label('out-of-sample AUC  (0.5 = no ranking)', color=INK_2, fontsize=8)
    cb.outline.set_visible(False)
    cb.ax.tick_params(color=GRID, labelsize=7.5)

    ax.set_xticks(np.arange(len(hz)) + 0.5)
    ax.set_xticklabels([f'+{h}{ws.horizon_unit}' for h in hz])
    ax.set_yticks(np.arange(len(th)) + 0.5)
    ax.set_yticklabels([_theta_pct(t, ws.theta_step) for t in th], fontsize=7.5)
    ax.tick_params(length=0)
    for side in ('top', 'right', 'left', 'bottom'):
        ax.spines[side].set_visible(False)
    ax.set_xlabel(f'horizon (+{ws.horizon_unit})')
    ax.set_ylabel('barrier θ')

    above = int(np.nansum(m > 0.5))
    tot = int(np.isfinite(m).sum())
    k = int(np.nanargmax(m))
    bi, bj = divmod(k, len(hz))
    _title(fig, 'Does the ranking survive, and where?',
           f'The same walk forward run at every target on the judged grid. Red ranks '
           f'better than chance out of sample, blue worse.\n'
           f'{above} of {tot} targets come out above 0.5. The strongest is '
           f'{_theta_pct(th[bi], ws.theta_step)} within +{int(hz[bj])}{ws.horizon_unit} at AUC '
           f'{m[bi, bj]:.2f} — modest, and that is what an honest out-of-sample number '
           f'on this question looks like.')
    _note(fig, f'{ws.dir.name} · one-node-per-family model, everything fit on training '
               f'windows only, scored on non-overlapping bars · blank cells had fewer '
               f'than {MIN_EVENTS} events on one side, where AUC turns on two or three '
               f'cases · AUC is unchanged by the scale correction, which moves '
               f'calibration and not order · 06_bayez.npz')
    fig.subplots_adjust(top=0.82)
    return _save(fig, out / '10_composition_grid.png')


# ── the stage ─────────────────────────────────────────────────────────────────

_CAPTIONS = {
    '01_band.png': 'The measured forward envelope starts at the last close.',
    '02_landscape.png': 'Magnitude clears the null; direction mostly does not.',
    '03_where_the_edge_is.png': 'Where the strongest cleared effects sit by horizon.',
    '04_families.png': 'Which feature families carry discoveries.',
    '05_null.png': 'The exact circular-shift null as a distribution, and its resolution '
                   'floor.',
    '06_funnel.png': 'The economic and statistical filters both matter.',
    '07_baseline_surface.png': 'The unconditional surface is the reference every '
                               'condition must beat.',
    '08_conditional_shift.png': 'One condition\'s strongest actionable deviation from '
                                'that baseline.',
    '09_composition.png': 'Composing conditions out of sample exposes overconfidence.',
    '10_composition_grid.png': 'Where the composed ranking survives out of sample.',
    '11_atr_regime_ladder.png': 'ATR turns one barrier into a visible risk ladder.',
    '12_null_gap_ranking.png': 'The strongest discoveries sit far beyond their own '
                               'null thresholds.',
}


def build(ws) -> None:
    """Render every figure this workspace has the artifacts for."""
    out = ws.result_dir
    tree = load_tree(ws.tree_path)
    evaluation = ws.read_json(ws.eval_path)
    cleared = ws.read_json(ws.cleared_path)

    if not cleared:
        print('No 05_gate.json - run --gate first.')
        return

    print(f"\n=== 7. Report [{ws.dir.name}] ===")
    print(f"  rendering from artifacts on disk; nothing here re-measures\n")

    # stale numbering from an earlier layout would otherwise linger beside the new files
    for stale in out.glob('*.png'):
        stale.unlink()

    jobs = [
        ('band',             lambda: fig_band(ws, tree, cleared, out)),
        ('landscape',        lambda: fig_landscape(ws, tree, evaluation, cleared, out)),
        ('where the edge is', lambda: fig_horizons(ws, cleared, out)),
        ('families',         lambda: fig_families(ws, tree, cleared, out)),
        ('the null',         lambda: fig_null(ws, tree, cleared, out)),
        ('funnel',           lambda: fig_funnel(ws, tree, evaluation, cleared, out)),
        ('baseline surface', lambda: fig_baseline(ws, out)),
        ('conditional shift', lambda: fig_shift(ws, tree, cleared, out)),
        ('composition',      lambda: fig_calibration(ws, out)),
        ('composition grid', lambda: fig_composition_grid(ws, out)),
        ('ATR ladder',       lambda: fig_atr_ladder(ws, cleared, out)),
        ('null gap ranking', lambda: fig_null_gap_ranking(ws, cleared, out)),
    ]

    written = []
    for name, fn in jobs:
        try:
            path = fn()
        except Exception as e:
            print(f"  {name:<20} [skip] {type(e).__name__}: {e}")
            continue
        if path is None:
            print(f"  {name:<20} [skip] artifact missing")
            continue
        written.append(path)
        print(f"  {name:<20} {path.name}")

    index = [f'# {ws.dir.name}', '',
             f"`{ws.asset['ticker']}` · {ws.asset.get('interval', '1d')} · "
             f"from {ws.start_date} · {len(all_nodes(tree))} nodes", '']
    for p in written:
        index += [f'### {_CAPTIONS.get(p.name, p.stem)}', '',
                  f'![{p.stem}]({p.name})', '']
    (out / 'README.md').write_text('\n'.join(index), encoding='utf-8')

    print(f"\n  wrote {len(written)} figures to workspaces/{ws.dir.name}/result/")
