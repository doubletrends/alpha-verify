"""Run-level diagnostic report figures."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from barrierlab.infrastructure import artifact_io
from barrierlab.presentation.report_style import (
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
from barrierlab.infrastructure.workspace import BASELINE_NODE
from barrierlab.presentation.workbooks import feature_label


def write_bin_score_null_histograms(ws, summary: dict) -> list[Path]:
    """Render each selected-bin comparison against its simulated-bin null."""
    out = ws.validation_summary_path.parent / "plot"
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for row in summary.get("tests", []):
        null = np.asarray(row.get("null_scores", []), dtype=float)
        null = null[np.isfinite(null)]
        if not len(null):
            continue
        observed = float(row["bin_score"])
        p95 = float(row["peak_p95"])
        fig, ax = plt.subplots(figsize=(7.2, 4.1))
        ax.hist(null, bins=min(20, max(8, len(null) // 5)), color="#d8d7d2",
                edgecolor=SURFACE, linewidth=0.6)
        ax.axvline(p95, color=INK_2, linewidth=1.2, linestyle="--", label="null p95")
        ax.axvline(observed, color=S1, linewidth=2.1, label="observed bin score")
        ax.set_xlabel("two-sided bin score (pp)")
        ax.set_ylabel("synthetic OHLC paths")
        _frame(ax, grid_axis="y")
        ax.legend(fontsize=8, frameon=False)
        _title(
            fig,
            f'{feature_label(row["node"])} bin {int(row["bin_number"])} — score vs synthetic null',
            f'observed {observed:.2f}pp · null p95 {p95:.2f}pp · '
            f'p={row["peak_p"]:.4f}, q={row.get("q_value", float("nan")):.4f}',
        )
        _note(fig, f'{ws.dir.name} · {len(null)} shared synthetic OHLC paths · '
                   f'external condition histories held fixed')
        fig.subplots_adjust(top=0.78, bottom=0.18)
        path = out / f'null_histogram__{row["rank"]:03d}__{row["node"]}__bin_{int(row["bin_number"]):02d}.png'
        paths.append(_save(fig, path))
    return paths


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
    ax.scatter(p95, y, s=62, color=INK_2, edgecolors=SURFACE,
               linewidths=1.4, zorder=3, label='null p95')
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
    ax.set_xlabel('peak |deviation| over the surface (%)')
    ax.set_xlim(0, float(np.nanmax(observed)) * 1.25)
    _frame(ax, grid_axis='x')
    ax.legend(loc='lower left', bbox_to_anchor=(1.02, 0), ncol=1,
              fontsize=8, labelcolor=INK_2, borderaxespad=0)

    at_floor = sum(1 for r in tests_top if r.get('at_floor'))
    _title(fig,
           f'Top effects sit {ratios[-1]:.1f}–{ratios[0]:.1f}x beyond their own null',
           f'Each row keeps one node: its strongest discovered horizon. Grey ticks are '
           f'the shuffled-null p95; blue dots are the observed peak. {at_floor} of these '
           f'{len(rows)} are already at the exact p-value floor.')
    _note(fig, f'{ws.dir.name} · ranked from 04_validation/validation.json tests · one best discovered '
               f'horizon per selected sheet')
    fig.subplots_adjust(top=1 - 1.12 / fig_h, left=0.25, right=0.76, bottom=0.14)
    return _save(fig, out / 'C_null_gap_ranking.png')


def _null_distribution_for_gate_row(ws, gate_row: dict) -> dict | None:
    """Load one cleared selected sheet's saved Stage 4 null distribution."""
    try:
        artifact = artifact_io.load_validation(ws.validation_array_path(gate_row))
    except (FileNotFoundError, ValueError):
        return None
    if "selected_peak_null" not in artifact:
        return None
    matches = np.flatnonzero(artifact["horizons"].astype(int) == int(gate_row["horizon"]))
    if len(matches) != 1:
        return None
    horizon_index = int(matches[0])
    null = np.asarray(artifact["selected_peak_null"][horizon_index], dtype=float)
    null = null[np.isfinite(null)]
    n_shifts = int(np.asarray(artifact["n_shifts"])[horizon_index])
    if not len(null) or n_shifts != len(null):
        return None
    observed = float(np.asarray(artifact["sheet_peak_real"])[0, horizon_index])
    return {
        "null": null,
        "observed": observed,
        "p95": float(np.percentile(null, 95)),
        "p_value": float((1.0 + (null >= observed).sum()) / (1.0 + n_shifts)),
        "n_shifts": n_shifts,
        "floor": 1.0 / (1.0 + n_shifts),
    }


def fig_null(ws, cleared, out: Path) -> Path | None:
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
        f"+{int(strong_row['horizon'])}{ws.horizon_unit}  {observed:.1f}%",
        (observed, ymax * 0.84),
        xytext=(8, 0),
        textcoords='offset points',
        color=S1,
        fontsize=8.8,
        fontweight='bold',
        ha='left',
        va='center',
    )
    ax.annotate(f'p99 {p99:.1f}%', (p99, ymax * 0.48),
                xytext=(-8, 0), textcoords='offset points', color=INK_2,
                fontsize=8, ha='right', va='center')

    ax.set_xlabel('peak |deviation| over each selected sheet (%)')
    ax.set_ylabel('density')
    ax.set_xlim(0, max(float(observed_all.max()), float(null.max())) * 1.12)
    ax.set_ylim(0, ymax * 1.15)
    _frame(ax, grid_axis='x')
    ax.legend(loc='lower left', bbox_to_anchor=(1.02, 0), ncol=1,
              fontsize=8, labelcolor=INK_2, borderaxespad=0)

    _title(fig, 'It is not luck: the null test rejects accidental alignment',
           f'Grey histogram pools the exact null draws for all {len(dists)} selected '
           f'and filtered sheets. The dashed line is the pooled p99 null threshold; '
           f'the blue tick is the strongest observed sheet peak.')
    _note(fig, f'A shift keeps the feature\'s autocorrelation and destroys only its '
               f'alignment with the future · histogram loaded from saved Stage 4 nulls · {n_shifts:,} pooled '
               f'usable shifts across {len(dists)} cleared sheets · strongest observed '
               f'p = {float(strongest["p_value"]):.2g}, per-sheet floor '
               f'{float(np.nanmin(floors)):.2g}')
    fig.subplots_adjust(top=0.74, right=0.64)
    return _save(fig, out / 'B_null.png')
