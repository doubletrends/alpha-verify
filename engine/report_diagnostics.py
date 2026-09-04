"""Run-level diagnostic report figures."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from engine import validate as val
from engine.report_common import headline_node as _headline_node
from engine.report_style import (
    CMAP_SEQ,
    FAINT,
    GRID,
    INK,
    INK_2,
    MUTED,
    ORD,
    S1,
    SEQ,
    SURFACE,
    frame as _frame,
    note as _note,
    plt,
    save as _save,
    title as _title,
)
from engine.writer import feature_label
from pipeline.runtime import loader, node_feature
from universe import all_nodes
from workspace import BASELINE_NODE


def fig_landscape(ws, universe, evaluation, cleared, out: Path) -> Path | None:
    """
    Every validated node as one point on the two axes that decide whether it clears.

    x is the economic effect size: the strongest qualifying cell's deviation from the
    baseline probability, in percentage points. y is the statistical strength: the
    observed surface peak divided by that node's own shuffled-null p95. The upper-right
    quadrant is the product claim: large enough to matter, and too strong to explain as
    feature alignment noise.
    """
    tests = cleared.get('tests', [])
    if not tests:
        return None

    best_null = {}
    for r in tests:
        if r.get('node') == BASELINE_NODE or not r.get('peak_p95'):
            continue
        v = float(r['peak_real']) / float(r['peak_p95'])
        cur = best_null.get(r['node'])
        if cur is None or v > cur[0]:
            best_null[r['node']] = (v, r)

    ev = evaluation.get('nodes', {})
    fam_of = {n['id']: n['family'] for n in all_nodes(universe)}
    cleared_nodes = {c['node'] for c in cleared.get('cleared', [])}

    pts = []
    for node, (ratio, row) in best_null.items():
        best = ev.get(node, {}).get('best')
        if not best:
            continue
        pts.append({
            'node': node,
            'ratio': ratio,
            'dev': abs(float(best['dev'])),
            'family': fam_of.get(node, '?'),
            'cleared': node in cleared_nodes,
            'horizon': int(row['horizon']),
        })
    if len(pts) < 5:
        return None

    xs = np.array([p['dev'] for p in pts])
    ys = np.array([p['ratio'] for p in pts])
    is_clear = np.array([p['cleared'] for p in pts], dtype=bool)

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.axvline(ws.min_dev, color=GRID, linewidth=1.0, zorder=1)
    ax.axhline(1.0, color=GRID, linewidth=1.0, zorder=1)
    ax.scatter(xs[~is_clear], ys[~is_clear], s=44, color=FAINT,
               edgecolors=SURFACE, linewidths=1.0, zorder=3, label='did not clear both')
    ax.scatter(xs[is_clear], ys[is_clear], s=70, c=xs[is_clear], cmap=CMAP_SEQ,
               vmin=ws.min_dev, vmax=max(float(xs.max()), ws.min_dev + 1),
               edgecolors=SURFACE, linewidths=1.4, zorder=5, label='cleared both')

    ax.set_xlabel('economic effect size: strongest qualifying deviation from baseline (pp)')
    ax.set_ylabel('statistical strength: observed surface peak ÷ shuffled-null p95')
    _frame(ax, grid_axis='both')
    ax.set_xlim(0, float(xs.max()) * 1.18)
    ax.set_ylim(0, float(ys.max()) * 1.16)
    ax.annotate('economic threshold', (ws.min_dev, ax.get_ylim()[1] * 0.92),
                xytext=(5, 0), textcoords='offset points', color=MUTED,
                fontsize=7.5, va='center')
    ax.annotate('null p95', (ax.get_xlim()[1] * 0.97, 1.0),
                xytext=(0, 4), textcoords='offset points', color=MUTED,
                fontsize=7.5, ha='right', va='bottom')
    ax.legend(loc='upper left', fontsize=8, labelcolor=INK_2)

    if is_clear.any():
        clear_idx = np.flatnonzero(is_clear)
        top = clear_idx[int(np.argmax(xs[is_clear]))]
        ax.annotate(f'strongest cleared\n{pts[top]["node"]}',
                    (float(xs[top]), float(ys[top])),
                    xytext=(-10, 16), textcoords='offset points', ha='right',
                    fontsize=7.8, fontweight='bold', color=INK,
                    arrowprops=dict(arrowstyle='-', color=MUTED, linewidth=0.8))

    n_econ = len(evaluation.get('passed', []))
    n_disc = len({t['node'] for t in tests if t.get('verdict') == 'discovery'})
    n_clear = int(is_clear.sum())
    _title(fig, 'Cleared both: useful size, null-tested strength',
           f'One point per validated node. Right of the vertical line means the best '
           f'cell passed the economic filter; above the horizontal line means the '
           f'surface peak beat its own shuffled-null p95. {n_clear} nodes clear both.')
    _note(fig, f'{ws.dir.name} · {n_econ} economically usable · {n_disc} nodes with a '
               f'Benjamini-Hochberg discovery · {n_clear} clear both · 04_gate.json + '
               f'evaluation.json')
    fig.subplots_adjust(top=0.78)
    return _save(fig, out / '02_landscape.png')


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
    _note(fig, f'{ws.dir.name} · ranked from 04_gate.json tests · one best discovered '
               f'horizon per node, so repeated variants do not fill the chart')
    fig.subplots_adjust(top=1 - 0.95 / fig_h, left=0.25, bottom=0.14)
    return _save(fig, out / '12_null_gap_ranking.png')


def fig_funnel(ws, universe, evaluation, cleared, out: Path) -> Path:
    n_nodes = len(all_nodes(universe))
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
    for (_, v, sub), y, colour in zip(stages, ys, ORD):
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
               f'04_gate.json')
    fig.subplots_adjust(top=0.74, left=0.19)
    return _save(fig, out / '06_funnel.png')


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
               '04_gate.json')
    fig.subplots_adjust(top=0.70)
    return _save(fig, out / '03_where_the_edge_is.png')


def fig_null(ws, universe, cleared, out: Path) -> Path | None:
    """
    The exact null as a distribution rather than a p-value.

    Every shift is a world in which the feature carries nothing but keeps its own
    autocorrelation. The observed surface either sits inside that cloud or it does not,
    and the width of the cloud is the honest answer to "how big is big".
    """
    head = _headline_node(ws, cleared, universe)
    if head is None:
        return None
    cube = barrier.load_cube(ws.summary_cube_path(head['family'], head['id']))
    h = int(head['gate']['horizon'])

    get_data = loader(ws)
    data, feat = node_feature(ws, head, get_data)
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


def fig_families(ws, universe, cleared, out: Path) -> Path | None:
    tests = cleared.get('tests', [])
    if not tests:
        return None
    fam_of = {n['id']: n['family'] for n in all_nodes(universe)}
    # the baseline family is excluded: shuffling a constant cannot move anything, so its
    # row is empty by construction and would read as a finding beside the ones that are
    fams = sorted({n['family'] for n in all_nodes(universe)} - {'_base'})
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
              f'{sum(1 for n in all_nodes(universe) if n["family"] in dead)} nodes between '
              f'them, and not one significant cell.' if dead else ''))
    _note(fig, f'{ws.dir.name} · 04_gate.json')
    fig.subplots_adjust(top=0.86 - 0.5 / len(fams), left=0.20, bottom=0.10)
    return _save(fig, out / '04_families.png')

