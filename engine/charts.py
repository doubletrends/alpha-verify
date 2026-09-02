"""
README figures, regenerated from committed artifacts.

Every figure is built from files already in the repo — the validation JSON written
by `--validate` and the node surfaces written by the compute pass — so
`python run.py --charts` reproduces the whole set offline, with no data fetch. The
skew panel additionally reads assets/skew_scores.json, a dated capture of `--skew`,
and is skipped if that file is absent.

Palette is deliberately flat: one accent blue, one warning amber, grey for
"indistinguishable from noise", and a muted red for the longest horizon.
Green/red on the heatmap follows the workbook convention (red = more risk).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
import openpyxl                   # noqa: E402

from engine import writer         # noqa: E402

ROOT   = Path(__file__).resolve().parent.parent
WS     = ROOT / 'workspaces'
ASSETS = ROOT / 'assets'

# ── palette / style ──────────────────────────────────────────────────────────
_INK     = '#333333'
_ACCENT  = '#2b6cb0'   # HAR / drawdown / "candidate structure"
_AMBER   = '#dd8f28'   # nominal significance
_MUTED   = '#a0aec0'   # noise, gridlines, naive baseline
_GREEN   = '#2f855a'   # survives Bonferroni
_RED     = '#b03a2e'   # implied vol benchmark / long horizon
_GRID    = '#e8e8e8'

_VERDICT = {
    'structure':    _GREEN,
    'nominal':      _AMBER,
    'noise':        _MUTED,
    'underpowered': _MUTED,
    'insufficient': _MUTED,
}
_MARK = {'pdf': '≈', 'above': '>', 'below': '<'}


def _apply_style() -> None:
    plt.rcParams.update({
        'figure.facecolor':   'white',
        'axes.facecolor':     'white',
        'axes.edgecolor':     _MUTED,
        'axes.labelcolor':    _INK,
        'axes.titlecolor':    _INK,
        'text.color':         _INK,
        'xtick.color':        _INK,
        'ytick.color':        _INK,
        'axes.grid':          True,
        'grid.color':         _GRID,
        'grid.linewidth':     0.6,
        'axes.spines.top':    False,
        'axes.spines.right':  False,
        'font.size':          10,
        'figure.dpi':         150,
        'savefig.dpi':        150,
        'savefig.bbox':       'tight',
    })


# ── loaders ──────────────────────────────────────────────────────────────────

def _load_validation(path: Path) -> tuple[dict, list[dict]]:
    """Return (raw dict, flat list of per-node-per-horizon rows)."""
    d = json.loads(Path(path).read_text(encoding='utf-8'))
    rows = []
    for nid, nd in d['nodes'].items():
        for hz, h in nd['horizons'].items():
            if h.get('real') is None or h.get('null_p95') is None:
                continue
            rows.append({
                'node':    nid,
                'family':  nd['family'],
                'horizon': hz,
                'real':    float(h['real']),
                'null_p95': float(h['null_p95']),
                'p_value': h.get('p_value'),
                'verdict': h.get('verdict', 'noise'),
            })
    return d, rows


def _read_surface(xlsx_path: Path, kind: str):
    """
    Read one sheet of a node surface workbook.

    Returns (thresholds, ns, horizon_labels, matrix) where thresholds is the list
    of parsed feature values (floats), ns the observation count per row, and matrix
    an (n_rows x n_horizons) array of pp deviations. `kind` is 'pdf' / 'above' / 'below'.
    """
    wb  = openpyxl.load_workbook(xlsx_path, data_only=True, read_only=True)
    ws  = writer.find_sheet(wb, kind)
    rows = list(ws.iter_rows(values_only=True))
    wb.close()

    hdr   = rows[4]
    hcols = [c for c in hdr if isinstance(c, str) and c.startswith('+')]
    mark  = _MARK[kind]

    thresholds, ns, mat = [], [], []
    for r in rows[5:]:
        if r[0] is None:
            break
        try:
            thresholds.append(float(str(r[0]).split(mark)[-1].strip()))
        except ValueError:
            continue
        ns.append(int(r[1]) if r[1] is not None else 0)
        mat.append([float(v) if isinstance(v, (int, float)) else np.nan
                    for v in r[2:2 + len(hcols)]])
    return thresholds, ns, hcols, np.array(mat, dtype=float)


# ── figures ──────────────────────────────────────────────────────────────────

BTC = WS / 'btc_daily_14days'
NDX = WS / 'nasdaq_hourly_24hrs'


def fig_peak_vs_null(out: Path) -> None:
    """Hero: measured peak deviation vs the noise ceiling, for both barriers."""
    _, down = _load_validation(BTC / 'validation.dd10.json')
    _, up_ = _load_validation(BTC / 'validation.run10.json')

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.7), sharex=True, sharey=True)
    panels = [
        (axes[0], down, 'Downside barrier:  touches −10% within h bars?'),
        (axes[1], up_, 'Upside barrier:  touches +10% within h bars?'),
    ]
    lim = 52
    for ax, rows, title in panels:
        ax.fill_between([0, lim], [0, lim], lim, color=_ACCENT, alpha=0.05, zorder=0)
        ax.plot([0, lim], [0, lim], color=_MUTED, lw=1, ls='--', zorder=1)
        for v in ('noise', 'nominal', 'structure'):
            pts = [(r['null_p95'], r['real']) for r in rows if r['verdict'] == v]
            if not pts:
                continue
            xs, ys = zip(*pts)
            ax.scatter(xs, ys, s=24, c=_VERDICT[v], edgecolors='white', linewidths=0.4,
                       label=f'{v}  ({len(pts)})', zorder=2 if v == 'noise' else 3)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('noise ceiling: 95th pct of shuffled null (pp)')
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.legend(frameon=False, fontsize=8, loc='upper left')
    axes[0].set_ylabel('measured peak deviation (pp)')
    fig.suptitle('Every condition, against the noise it has to beat', fontsize=12, fontweight='bold')
    fig.text(0.5, -0.03,
             'One point per feature × horizon, BTC daily. On or below the dashed line = '
             'a shuffle of the same data reaches that peak just as often.',
             ha='center', fontsize=8, color=_MUTED)
    fig.savefig(out)
    plt.close(fig)


def fig_nominal_hits(out: Path) -> None:
    """Hits clearing p<0.05 per workspace x barrier, against chance."""
    sets = [
        ('Down −10%\nBTC daily', BTC / 'validation.dd10.json', _ACCENT),
        ('Up +10%\nBTC daily', BTC / 'validation.run10.json', _MUTED),
        ('Down −2%\nNASDAQ hourly', NDX / 'validation.dd2.json', _ACCENT),
        ('Up +2%\nNASDAQ hourly', NDX / 'validation.run2.json', _MUTED),
    ]
    data = []
    for label, path, color in sets:
        if not path.exists():
            continue
        d, _ = _load_validation(path)
        data.append((label, d, color))
    if not data:
        return

    fig, ax = plt.subplots(figsize=(9, 4.7))
    top = max(d['summary']['nominal'] for _, d, _ in data)
    for i, (label, d, color) in enumerate(data):
        nominal = d['summary']['nominal']
        struct = d['summary']['structure']
        expected = 0.05 * d['n_tests']
        ax.bar(i, nominal, width=0.55, color=color, zorder=3)
        ax.hlines(expected, i - 0.32, i + 0.32, color=_INK, lw=1.6, ls='--', zorder=4)
        ax.text(i, nominal + 1.4, str(nominal), ha='center', fontweight='bold')
        if struct:
            ax.text(i, nominal + 5.0, f'{struct} clear\nBonferroni', ha='center',
                    fontsize=8, color=_GREEN)
        ax.text(i, expected + top * 0.022, f'{expected:.0f} by chance', ha='center',
                va='bottom', fontsize=7.5, color=_INK)
    ax.set_xticks(range(len(data)))
    ax.set_xticklabels([d[0] for d in data])
    ax.set_ylabel('nodes significant at p < 0.05')
    ax.set_ylim(0, top * 1.35 + 8)
    ax.set_title('Same features, same machinery — only the barrier changes', fontweight='bold')
    fig.text(0.5, -0.02,
             'Dashed line = false positives expected from testing this many nodes at α = 0.05.',
             ha='center', fontsize=8, color=_MUTED)
    fig.savefig(out)
    plt.close(fig)


def fig_touch_surface(out: Path) -> None:
    """Heatmap of one node's conditional touch-probability surface."""
    path = BTC / 'volatility' / 'dd10' / 'bb_pct_20.xlsx'
    thr, ns, hcols, mat = _read_surface(path, 'pdf')

    fig, ax = plt.subplots(figsize=(9, 6.2))
    vlim = 28
    im = ax.imshow(mat, aspect='auto', cmap='RdYlGn_r', vmin=-vlim, vmax=vlim)
    ax.set_xticks(range(len(hcols)))
    ax.set_xticklabels(hcols, fontsize=8)
    ax.set_yticks(range(len(thr)))
    ax.set_yticklabels([f'{v:+.2f}   n={n}' for v, n in zip(thr, ns)], fontsize=7)
    ax.set_xlabel('forecast horizon')
    ax.set_ylabel('Bollinger %b  (feature value — low = price pinned to the lower band)')
    ax.set_title("A single node's surface:  P(touch −10% | %b ≈ x) − base rate",
                 fontweight='bold')
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
    cb.set_label('deviation from base rate (pp)', fontsize=8)
    fig.text(0.5, 0.01,
             'The engine writes one of these per feature per barrier. The dark band is the '
             'readable signal; everything pale is within noise.',
             ha='center', fontsize=8, color=_MUTED)
    fig.savefig(out)
    plt.close(fig)


def fig_drawdown_monotone(out: Path) -> None:
    """The overextension to downside-touch signal, monotone across thresholds."""
    path = BTC / 'ma' / 'dd10' / 'ma_ratio_200.xlsx'
    thr, ns, hcols, mat = _read_surface(path, 'above')
    thr = np.array(thr)

    fig, ax = plt.subplots(figsize=(8, 4.9))
    for hz, color, lw in [('+3d', _MUTED, 1.3), ('+7d', _ACCENT, 1.7), ('+14d', _RED, 2.1)]:
        if hz not in hcols:
            continue
        j = hcols.index(hz)
        ax.plot(thr, mat[:, j], color=color, lw=lw, marker='o', ms=3, label=hz)
    ax.axhline(0, color=_INK, lw=0.8)

    j14 = hcols.index('+14d')
    ax.annotate(
        f'price > {thr[-1] * 100:.0f}% above its 200-day average\n'
        f'→ +{mat[-1, j14]:.0f} pp at +14d   (n = {ns[-1]})',
        xy=(thr[-1], mat[-1, j14]),
        xytext=(thr[max(0, len(thr) // 3)], mat[-1, j14] - 6),
        fontsize=8, color=_INK,
        arrowprops=dict(arrowstyle='->', color=_MUTED),
    )
    ax.set_xlabel('threshold  τ   (condition:  price / 200-day MA − 1  >  τ)')
    ax.set_ylabel('P(touch −10%) − base rate (pp)')
    ax.set_title('Overextension precedes downside touches, monotonically', fontweight='bold')
    ax.legend(frameon=False, fontsize=9, title='horizon')
    fig.savefig(out)
    plt.close(fig)


def fig_skew(out: Path) -> None:
    """
    The question the endpoint framing could not ask: does a condition tilt downside,
    or does it just raise both barriers?

    Each node is plotted as its peak downside deviation against the upside deviation
    measured on the same bars, at the same slice of feature space. Points on the
    diagonal move both barriers equally — a volatility proxy carrying no directional
    information. Distance below the diagonal is the genuinely asymmetric part.
    """
    path = ASSETS / 'skew_scores.json'
    if not path.exists():
        return
    payload = json.loads(path.read_text(encoding='utf-8'))
    rows = payload['nodes']
    if not rows:
        return

    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    lim = max(max(abs(r['down_dev']) for r in rows),
              max(abs(r['up_dev']) for r in rows)) * 1.12

    ax.fill_between([-lim, lim], [-lim, lim], -lim, color=_ACCENT, alpha=0.05, zorder=0)
    ax.plot([-lim, lim], [-lim, lim], color=_MUTED, lw=1, ls='--', zorder=1)
    ax.axhline(0, color=_GRID, lw=1)
    ax.axvline(0, color=_GRID, lw=1)

    for r in rows:
        c = _ACCENT if r['skew'] > 5 else (_AMBER if r['skew'] < -5 else _MUTED)
        ax.scatter(r['down_dev'], r['up_dev'], s=34, c=c,
                   edgecolors='white', linewidths=0.5, zorder=3)
    # Label the extremes only, and declutter: points close together get their labels
    # pushed apart vertically so the annotations stay readable.
    labelled = sorted(rows, key=lambda r: -abs(r['skew']))[:6]
    placed: list[tuple[float, float]] = []
    for r in labelled:
        x, y = r['down_dev'], r['up_dev']
        dy = -3.0
        # nudge until this label is clear of every one already placed
        while any(abs(x - px) < lim * 0.30 and abs((y + dy * lim / 90) - py) < lim * 0.055
                  for px, py in placed):
            dy -= 11.0
        placed.append((x, y + dy * lim / 90))
        ax.annotate(r['node'], (x, y), fontsize=7.5,
                    xytext=(6, dy), textcoords='offset points', color=_INK,
                    arrowprops=dict(arrowstyle='-', color=_MUTED, lw=0.5,
                                    shrinkA=0, shrinkB=2) if dy < -6 else None)

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel('P(touch −10%) deviation at the peak bin (pp)')
    ax.set_ylabel('P(touch +10%) deviation, same bars, same bin (pp)')
    ax.set_title('Real asymmetry, or just volatility?', fontweight='bold')
    ax.text(0.97, 0.06, 'on the diagonal =\nraises both barriers equally',
            transform=ax.transAxes, ha='right', fontsize=8, color=_MUTED)
    fig.text(0.5, -0.01,
             f"Below the diagonal = downside-specific. BTC daily, {payload.get('horizon', '+7d')}, "
             f'one point per node.',
             ha='center', fontsize=8, color=_MUTED)
    fig.savefig(out)
    plt.close(fig)


_FIGURES = [
    ('peak_vs_null.png',      fig_peak_vs_null),
    ('nominal_hits.png',      fig_nominal_hits),
    ('touch_surface.png',     fig_touch_surface),
    ('drawdown_monotone.png', fig_drawdown_monotone),
    ('barrier_skew.png',      fig_skew),
]


def render_all(out_dir: str | Path | None = None) -> Path:
    """Regenerate every README figure into out_dir (default: repo assets/)."""
    out = Path(out_dir) if out_dir else ASSETS
    out.mkdir(parents=True, exist_ok=True)
    _apply_style()
    for name, fn in _FIGURES:
        try:
            fn(out / name)
            print(f'  wrote {out / name}')
        except FileNotFoundError as e:
            print(f'  [charts] skip {name} — missing input: {e}')
    return out
