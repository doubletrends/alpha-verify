"""
README figures, regenerated from committed artifacts.

Every figure is built from files already in the repo — the validation JSON written
by `--validate` and the node surfaces written by the compute pass — so
`python run.py --charts` reproduces the whole set offline, with no data fetch. The
sole exception is the volatility-forecast panel, which needs live implied-vol data;
it is drawn from a dated capture in assets/volforecast_scores.json and skipped if
that file is absent.

Palette is deliberately flat: one accent blue, one warning amber, grey for
"indistinguishable from noise", and a muted red for the option-market benchmark.
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

def fig_peak_vs_null(out: Path) -> None:
    """Hero: measured peak deviation vs the noise ceiling, direction | drawdown."""
    _, direction = _load_validation(WS / 'btc_daily_14days' / 'validation.json')
    _, drawdown  = _load_validation(WS / 'btc_daily_14days' / 'validation.dd10.json')

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.7), sharex=True, sharey=True)
    panels = [
        (axes[0], direction, 'Direction:  higher in h bars?'),
        (axes[1], drawdown,  'Drawdown:  down ≥10% within h bars?'),
    ]
    lim = 48
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
    fig.suptitle('Every signal, against the noise it has to beat', fontsize=12, fontweight='bold')
    fig.text(0.5, -0.03,
             'One point per feature × horizon (195 each, BTC daily). On or below the dashed line = '
             'a shuffle of the data reaches the same peak just as often.',
             ha='center', fontsize=8, color=_MUTED)
    fig.savefig(out)
    plt.close(fig)


def fig_nominal_hits(out: Path) -> None:
    """How many nodes clear p<0.05, per workspace/outcome, vs chance."""
    dir_d, _ = _load_validation(WS / 'btc_daily_14days' / 'validation.json')
    ndx_d, _ = _load_validation(WS / 'nasdaq_hourly_24hrs' / 'validation.json')
    dd_d, _  = _load_validation(WS / 'btc_daily_14days' / 'validation.dd10.json')

    data = [
        ('Direction\nBTC daily',     dir_d, _MUTED),
        ('Direction\nNASDAQ hourly', ndx_d, _MUTED),
        ('Drawdown ≥10%\nBTC daily', dd_d, _ACCENT),
    ]
    fig, ax = plt.subplots(figsize=(8, 4.7))
    x = np.arange(len(data))
    for i, (label, d, color) in enumerate(data):
        n_tests  = d['n_tests']
        nominal  = d['summary']['nominal']
        struct   = d['summary']['structure']
        expected = 0.05 * n_tests
        ax.bar(i, nominal, width=0.55, color=color, zorder=3)
        ax.hlines(expected, i - 0.32, i + 0.32, color=_INK, lw=1.6, ls='--', zorder=4)
        ax.text(i, nominal + 1.4, str(nominal), ha='center', fontweight='bold')
        if struct:
            ax.text(i, nominal + 5.2, f'{struct} clear\nBonferroni', ha='center',
                    fontsize=8, color=_GREEN)
        ax.text(i + 0.36, expected, f'  {expected:.0f} by chance', va='center',
                fontsize=8, color=_INK)
    ax.set_xticks(x)
    ax.set_xticklabels([d[0] for d in data])
    ax.set_ylabel('nodes significant at p < 0.05')
    ax.set_ylim(0, 72)
    ax.set_title('Same features, same machinery — only the question changes', fontweight='bold')
    fig.text(0.5, -0.02,
             'Dashed line = false positives expected from testing this many nodes at α = 0.05.',
             ha='center', fontsize=8, color=_MUTED)
    fig.savefig(out)
    plt.close(fig)


def fig_winrate_surface(out: Path) -> None:
    """Heatmap of one node's conditional-probability surface."""
    path = WS / 'btc_daily_14days' / 'volatility' / 'dd10' / 'bb_pct_20.xlsx'
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
    ax.set_title("A single node's surface:  P(10% drawdown | %b ≈ x) − base rate",
                 fontweight='bold')
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
    cb.set_label('deviation from 21.7% base rate (pp)', fontsize=8)
    fig.text(0.5, 0.01,
             'The engine writes one of these for every feature. The dark band is the readable signal; '
             'everything pale is within noise.', ha='center', fontsize=8, color=_MUTED)
    fig.savefig(out)
    plt.close(fig)


def fig_drawdown_monotone(out: Path) -> None:
    """The overextension → drawdown signal, monotone across thresholds."""
    path = WS / 'btc_daily_14days' / 'ma' / 'dd10' / 'ma_ratio_200.xlsx'
    thr, ns, hcols, mat = _read_surface(path, 'above')
    thr = np.array(thr)

    fig, ax = plt.subplots(figsize=(8, 4.9))
    series = [('+3d', _MUTED, 1.3), ('+7d', _ACCENT, 1.7), ('+14d', _RED, 2.1)]
    for hz, color, lw in series:
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
    ax.set_ylabel('P(10% drawdown) − base rate (pp)')
    ax.set_title('The signal that holds up: overextension precedes drawdowns, monotonically',
                 fontweight='bold')
    ax.legend(frameon=False, fontsize=9, title='horizon')
    fig.savefig(out)
    plt.close(fig)


def fig_volforecast(out: Path) -> None:
    """HAR-RV vs naive persistence vs implied vol (dated capture)."""
    path = ASSETS / 'volforecast_scores.json'
    if not path.exists():
        print('  [charts] skip volforecast.png — assets/volforecast_scores.json missing')
        return
    d  = json.loads(path.read_text(encoding='utf-8'))
    hs = list(d['horizons'])
    models = [
        ('har',      'HAR-RV',                    _ACCENT),
        ('rv_naive', 'naive (carry RV forward)',  _MUTED),
        ('iv',       'implied vol (DVOL)',        _RED),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    metrics = [
        (axes[0], 'qlike',  'QLIKE  (lower is better)'),
        (axes[1], 'r2_log', 'out-of-sample R² in logs  (higher is better)'),
    ]
    for ax, metric, label in metrics:
        x = np.arange(len(hs))
        w = 0.26
        for k, (mk, mlabel, c) in enumerate(models):
            vals = [d['horizons'][h]['models'][mk][metric] for h in hs]
            ax.bar(x + (k - 1) * w, vals, width=w, color=c, label=mlabel, zorder=3)
        ax.axhline(0, color=_INK, lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([f'h = {h}' for h in hs])
        ax.set_title(label, fontsize=10)
    axes[0].legend(frameon=False, fontsize=8)

    n_lo = d['horizons'][hs[0]]['n']
    n_hi = d['horizons'][hs[-1]]['n']
    fig.suptitle('Volatility forecast: HAR-RV clears the naive baseline, ties the option market',
                 fontsize=12, fontweight='bold')
    fig.text(0.5, -0.04,
             f'BTC, {n_hi:,}–{n_lo:,} walk-forward forecasts · captured {d["captured"]} · '
             'encompassing test: implied vol absorbs HAR (β_HAR ≈ 0)',
             ha='center', fontsize=8, color=_MUTED)
    fig.savefig(out)
    plt.close(fig)


# ── driver ───────────────────────────────────────────────────────────────────

_FIGURES = [
    ('peak_vs_null.png',      fig_peak_vs_null),
    ('nominal_hits.png',      fig_nominal_hits),
    ('winrate_surface.png',   fig_winrate_surface),
    ('drawdown_monotone.png', fig_drawdown_monotone),
    ('volforecast.png',       fig_volforecast),
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
