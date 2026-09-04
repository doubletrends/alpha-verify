"""Run-level diagnostic report figures."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from engine import barrier, validate as val
from engine.report_common import headline_node as _headline_node
from engine.report_style import (
    FAINT,
    INK,
    INK_2,
    MUTED,
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
from workspace import BASELINE_NODE


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

