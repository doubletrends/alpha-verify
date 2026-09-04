"""Run-level diagnostic report figures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from engine import selection, shift, validate as val
from engine.report_style import (
    INK,
    INK_2,
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
        key = (r['node'], int(r.get('bin', -1)))
        cur = best.get(key)
        if cur is None or ratio > cur[0]:
            best[key] = (ratio, r)
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

    labels = [f'{feature_label(r["node"])} D{int(r.get("bin_number", 0))}  +{r["horizon"]}{ws.horizon_unit}'
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
               f'horizon per selected sheet')
    fig.subplots_adjust(top=1 - 0.95 / fig_h, left=0.25, bottom=0.14)
    return _save(fig, out / '12_null_gap_ranking.png')


def _null_distribution_for_gate_row(ws, gate_row: dict) -> dict | None:
    """Recompute one cleared selected sheet's null distribution from saved arrays."""
    b = int(gate_row.get('bin', gate_row.get('best_cell', {}).get('bin', 0)))
    sel = ws.read_json(ws.selection_path).get('selected', [])
    row = next((x for x in sel
                if x.get('node') == gate_row['node'] and int(x.get('bin', -1)) == b), None)
    artifact = None
    if row and ws.has_selection_array(row):
        artifact = selection.load_sheet(ws.selection_array_path(row))
    elif ws.shift_cube_path(gate_row['family'], gate_row['node']).exists():
        artifact = shift.load(ws.shift_cube_path(gate_row['family'], gate_row['node']))

    if artifact is None or any(k not in artifact for k in ('feature_values', 'high', 'low', 'close')):
        return None

    data = pd.DataFrame(
        {k: artifact[k].astype(float) for k in ('high', 'low', 'close')},
        index=pd.to_datetime(artifact['index']) if 'index' in artifact else None,
    )
    feat = pd.Series(artifact['feature_values'].astype(float), index=data.index)
    source_bin = int(artifact['source_bin']) if 'source_bin' in artifact else b
    return val.peak_shift_distribution(
        data,
        feat,
        int(gate_row['horizon']),
        artifact['thetas'],
        artifact['edges'],
        bin_index=source_bin,
        min_n=ws.min_bin_n,
    )


def fig_null(ws, universe, cleared, out: Path) -> Path | None:
    """The pooled exact null distribution behind every cleared selected sheet."""
    rows = [r for r in cleared.get('cleared', [])
            if r.get('best_cell') and r.get('node') != BASELINE_NODE]
    if not rows:
        return None

    dists = []
    for row in rows:
        try:
            dist = _null_distribution_for_gate_row(ws, row)
        except Exception:
            continue
        if not dist:
            continue
        null = np.asarray(dist['null'], dtype=float)
        null = null[np.isfinite(null)]
        if null.size >= 50:
            dists.append({**dist, 'null': null, 'row': row})
    if not dists:
        return None

    null = np.concatenate([d['null'] for d in dists])
    observed_all = np.array([float(d['observed']) for d in dists])
    floors = np.array([float(d['floor']) for d in dists])
    n_shifts = int(sum(int(d['n_shifts']) for d in dists))
    strongest = dists[int(np.nanargmax(observed_all))]
    strong_row = strongest['row']
    observed = float(strongest['observed'])
    p99 = float(np.percentile(null, 99))

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    counts, edges, _ = ax.hist(
        null,
        bins=64,
        density=True,
        color='#d8d7d2',
        edgecolor=SURFACE,
        linewidth=0.55,
        label='pooled circular-shift null',
    )
    ymax = float(np.nanmax(counts)) if np.isfinite(counts).any() else 1.0
    ax.axvline(p99, color=INK_2, linewidth=1.5, linestyle='--', label='pooled null p99')

    ax.axvline(observed, color=S1, linewidth=2.3, label='strongest observed')
    ax.annotate(
        f"{feature_label(strong_row['node'])} D{int(strong_row.get('bin_number', 0))} "
        f"+{int(strong_row['horizon'])}{ws.horizon_unit}  {observed:.1f} pp",
        (observed, ymax * 0.84),
        xytext=(8, 0),
        textcoords='offset points',
        color=S1,
        fontsize=8.8,
        fontweight='bold',
        ha='left',
        va='center',
    )
    ax.annotate(f'p99 {p99:.1f} pp', (p99, ymax * 0.48),
                xytext=(-8, 0), textcoords='offset points', color=INK_2,
                fontsize=8, ha='right', va='center')

    ax.set_xlabel('peak |deviation| over each selected sheet (pp)')
    ax.set_ylabel('density')
    ax.set_xlim(0, max(float(observed_all.max()), float(null.max())) * 1.12)
    ax.set_ylim(0, ymax * 1.15)
    _frame(ax, grid_axis='x')
    ax.legend(loc='upper right', fontsize=8, labelcolor=INK_2)

    _title(fig, 'The cleared sheets sit beyond the shuffled null',
           f'Grey histogram pools the exact null draws for all {len(dists)} selected '
           f'and filtered sheets. The dashed line is the pooled p99 null threshold; '
           f'the blue tick is the strongest observed sheet peak.')
    _note(fig, f'A shift keeps the feature\'s autocorrelation and destroys only its '
               f'alignment with the future · histogram recomputed in Stage 7 from saved '
               f'03_selection_array / 02_shift_array histories · {n_shifts:,} pooled '
               f'usable shifts across {len(dists)} cleared sheets · strongest observed '
               f'p = {float(strongest["p_value"]):.2g}, per-sheet floor '
               f'{float(np.nanmin(floors)):.2g}')
    fig.subplots_adjust(top=0.78)
    return _save(fig, out / '05_null.png')

