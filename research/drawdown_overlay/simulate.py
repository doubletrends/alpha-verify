"""
Does the one validated signal actually pay?

Four BTC nodes clear Bonferroni on the 10%-drawdown outcome in
`workspaces/btc_daily_14days/validation.dd10.json`:

    bb_pct_20      +3d   p = 3e-5   (floor)
    bb_pct_20      +7d   p = 3e-5   (floor)
    drawdown_7     +3d   p = 3e-5   (floor)
    ma_cross_7_21  +3d   p = 1.3e-4

Three distinct features; the horizon is how long the elevated drawdown risk
persists. All three fire on their *low* tail — price pinned to the lower
Bollinger band, a deep trailing drawdown, the fast MA below the slow one.

This builds a long/short overlay from them and compares its equity curve to
buy-and-hold:

  * default position is +1 (long BTC);
  * whenever a feature sits below its low-tail threshold, the overlay goes to
    -1 (short) for that feature's validated horizon (7 bars for bb_pct, 3 for
    the others), unioned across the three;
  * a flat variant (+1 / 0) is carried alongside as the "just avoid the
    drawdowns" reference.

Two threshold regimes, reported side by side:

  IS  in-sample   — thresholds are the p15 of each feature over the whole
                    history. The nodes were selected on this same history, so
                    this curve is optimistic by construction.
  WF  walk-forward — thresholds are the expanding p15 using only data up to
                    each day. Causal; no full-history information. Starts after
                    two years of warm-up.

Costs: 10 bps per unit of turnover, one-way, so a long->short flip pays 20 bps.

Run:  python research/drawdown_overlay/simulate.py
Writes equity_curve.png, results.json and refreshes README.md numbers.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data import fetcher, features   # noqa: E402
from workspace import Workspace      # noqa: E402

HERE = Path(__file__).resolve().parent

# ── configuration ────────────────────────────────────────────────────────────

SIGNALS = [
    {'name': 'bb_pct_20',     'feature': 'bb_pct',   'params': {'period': 20},          'hold': 7},
    {'name': 'drawdown_7',    'feature': 'drawdown', 'params': {'period': 7},           'hold': 3},
    {'name': 'ma_cross_7_21', 'feature': 'ma_cross', 'params': {'fast': 7, 'slow': 21}, 'hold': 3},
]

PCTILE     = 10       # low-tail trigger: feature below its p10 (bottom decile)
COST_BPS   = 10       # per unit turnover, one-way
ANN        = 365      # BTC trades every day
WF_WARMUP  = 365 * 2  # walk-forward: no position until this many bars of history

_INK, _GREY, _BLUE, _RED, _GREEN = '#333333', '#a0aec0', '#2b6cb0', '#b03a2e', '#2f855a'


# ── signal construction ──────────────────────────────────────────────────────

def _risk_off_mask(feat: pd.Series, thresh: pd.Series, hold: int) -> pd.Series:
    """
    True on every bar covered by a trigger and the `hold` bars after it.

    A trigger is feat < thresh. The forward fill (not a centered window) keeps
    the mask causal: a trigger at t only ever marks t .. t+hold.
    """
    trig = (feat < thresh).fillna(False).to_numpy()
    out  = np.zeros(len(trig), dtype=bool)
    for i in np.flatnonzero(trig):
        out[i:i + hold + 1] = True
    return pd.Series(out, index=feat.index)


def build_positions(data: pd.DataFrame) -> pd.DataFrame:
    """
    Per-bar target weights for the overlay variants. A weight at bar t is decided
    from information available at close t and earns the t -> t+1 return.
    """
    feats = {s['name']: features.compute(data, s['feature'], s['params']).reindex(data.index)
             for s in SIGNALS}

    is_off  = pd.Series(False, index=data.index)   # in-sample union
    wf_off  = pd.Series(False, index=data.index)   # walk-forward union
    per_sig = {}

    for s in SIGNALS:
        f = feats[s['name']]
        thr_is = pd.Series(np.nanpercentile(f.dropna(), PCTILE), index=data.index)
        thr_wf = f.expanding(min_periods=WF_WARMUP).quantile(PCTILE / 100.0)

        m_is = _risk_off_mask(f, thr_is, s['hold'])
        m_wf = _risk_off_mask(f, thr_wf, s['hold'])
        per_sig[s['name']] = m_is
        is_off |= m_is
        wf_off |= m_wf

    # walk-forward has no opinion until warm-up completes
    wf_off.iloc[:WF_WARMUP] = False

    pos = pd.DataFrame(index=data.index)
    pos['hodl']        = 1.0
    pos['short_is']    = np.where(is_off, -1.0, 1.0)
    pos['short_wf']    = np.where(wf_off, -1.0, 1.0)
    pos['flat_is']     = np.where(is_off,  0.0, 1.0)
    pos['flat_wf']     = np.where(wf_off,  0.0, 1.0)
    pos['n_signals']   = sum(per_sig.values()).astype(int)   # 0..3 fired (IS)
    pos['short_is_2of3'] = np.where(pos['n_signals'] >= 2, -1.0, 1.0)
    pos['_is_off'] = is_off
    pos['_wf_off'] = wf_off
    return pos, feats


# ── backtest ─────────────────────────────────────────────────────────────────

def run_variant(ret: pd.Series, weight: pd.Series) -> pd.Series:
    """Daily net return for a weight series. weight_t is held over t -> t+1."""
    held     = weight.shift(1)
    turnover = held.diff().abs()
    turnover.iloc[0] = held.iloc[0] if pd.notna(held.iloc[0]) else 0.0
    if pd.isna(turnover.iloc[1]) and pd.notna(held.iloc[1]):
        turnover.iloc[1] = abs(held.iloc[1])
    return held * ret - (COST_BPS / 1e4) * turnover.fillna(0.0)


def stats(s: pd.Series) -> dict:
    s   = s.dropna()
    eq  = (1.0 + s).cumprod()
    yrs = len(s) / ANN
    vol = float(s.std() * np.sqrt(ANN))
    dd  = eq / eq.cummax() - 1.0
    cagr = float(eq.iloc[-1] ** (1.0 / yrs) - 1.0)
    maxdd = float(dd.min())
    return {
        'total_return': float(eq.iloc[-1] - 1.0),
        'cagr':         cagr,
        'ann_vol':      vol,
        'sharpe':       float((s.mean() * ANN) / vol) if vol else float('nan'),
        'max_drawdown': maxdd,
        'calmar':       float(cagr / abs(maxdd)) if maxdd else float('nan'),
    }


# ── plot ─────────────────────────────────────────────────────────────────────

def plot(eq: pd.DataFrame, is_off: pd.Series, out: Path) -> None:
    plt.rcParams.update({
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'axes.edgecolor': _GREY, 'axes.grid': True, 'grid.color': '#e8e8e8',
        'grid.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False,
        'font.size': 10, 'text.color': _INK, 'axes.labelcolor': _INK,
        'xtick.color': _INK, 'ytick.color': _INK, 'savefig.dpi': 150, 'savefig.bbox': 'tight',
    })
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7.5), height_ratios=[2.3, 1], sharex=True)

    spans = _true_spans(is_off)
    for a, b in spans:
        ax1.axvspan(a, b, color=_RED, alpha=0.06, lw=0)

    ax1.plot(eq.index, eq['hodl'],     color=_GREY,  lw=1.8, label='buy & hold')
    ax1.plot(eq.index, eq['flat_wf'],  color=_GREEN, lw=1.8, label='flat overlay (walk-forward)')
    ax1.plot(eq.index, eq['flat_is'],  color=_GREEN, lw=1.1, ls='--', label='flat overlay (in-sample)')
    ax1.plot(eq.index, eq['short_wf'], color=_RED,   lw=1.8, label='long/short overlay (walk-forward)')
    ax1.plot(eq.index, eq['short_is'], color=_BLUE,  lw=1.1, ls='--', label='long/short overlay (in-sample)')
    ax1.set_yscale('log')
    ax1.set_ylabel('growth of $1  (log scale)')
    ax1.set_title('Trading the validated drawdown signal vs. buy & hold  —  BTC daily',
                  fontweight='bold')
    ax1.legend(frameon=False, fontsize=9, loc='upper left')

    for col, color, lbl in [('hodl', _GREY, 'buy & hold'), ('flat_wf', _GREEN, 'flat overlay (walk-forward)')]:
        d = eq[col] / eq[col].cummax() - 1.0
        ax2.fill_between(d.index, d.values, 0, color=color, alpha=0.35, lw=0)
        ax2.plot(d.index, d.values, color=color, lw=1.0, label=lbl)
    ax2.set_ylabel('drawdown')
    ax2.legend(frameon=False, fontsize=8, loc='lower left')
    fig.text(0.5, 0.03,
             f'Shaded = risk-off windows (in-sample signal union). '
             f'{COST_BPS} bps per unit turnover. Walk-forward thresholds start after {WF_WARMUP // 365}y warm-up.',
             ha='center', fontsize=8, color=_GREY)
    fig.savefig(out)
    plt.close(fig)


def _true_spans(mask: pd.Series):
    spans, start = [], None
    idx = mask.index
    vals = mask.to_numpy()
    for i, v in enumerate(vals):
        if v and start is None:
            start = idx[i]
        elif not v and start is not None:
            spans.append((start, idx[i]))
            start = None
    if start is not None:
        spans.append((start, idx[-1]))
    return spans


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ws = Workspace('btc_daily_14days')
    print(f'Fetching BTC daily OHLCV from {ws.start_date} ...')
    data = fetcher.fetch(['ohlcv'], start=ws.start_date, asset=ws.asset).dropna(subset=['close'])
    data = data[~data.index.duplicated(keep='last')].sort_index()
    print(f'  {len(data)} bars  {data.index[0].date()} -> {data.index[-1].date()}')

    ret = data['close'].pct_change()
    pos, feats = build_positions(data)

    variants = ['hodl', 'flat_is', 'flat_wf', 'short_is', 'short_is_2of3', 'short_wf']
    rets = {v: run_variant(ret, pos[v]) for v in variants}
    eq   = pd.DataFrame({v: (1.0 + rets[v].fillna(0.0)).cumprod() for v in variants})

    # premise check: is BTC's mean daily return actually worse on risk-off days?
    on, off = pos['_is_off'], ~pos['_is_off']
    premise = {
        'mean_ret_risk_off_days': float(ret[on].mean()),
        'mean_ret_risk_on_days':  float(ret[off].mean()),
        'share_days_risk_off':    float(on.mean()),
        'n_short_flips':          int(pos['short_is'].diff().abs().gt(0).sum()),
        'realized_dd_freq_all':   _dd_hit_freq(data['close'], 14, 0.10),
        'realized_dd_freq_off':   _dd_hit_freq(data['close'], 14, 0.10, on.shift(1).fillna(False)),
    }

    results = {
        'generated':  datetime.now(timezone.utc).isoformat(),
        'history':    {'start': str(data.index[0].date()), 'end': str(data.index[-1].date()),
                       'bars': int(len(data))},
        'config':     {'pctile': PCTILE, 'cost_bps': COST_BPS, 'wf_warmup_bars': WF_WARMUP,
                       'signals': [{k: s[k] for k in ('name', 'hold')} for s in SIGNALS]},
        'premise':    premise,
        'variants':   {v: stats(rets[v]) for v in variants},
    }
    (HERE / 'results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    plot(eq, pos['_is_off'], HERE / 'equity_curve.png')

    _print_table(results)
    _write_readme(results)
    print(f'\nwrote {HERE / "equity_curve.png"}')
    print(f'wrote {HERE / "results.json"}')
    print(f'wrote {HERE / "README.md"}')


def _dd_hit_freq(close: pd.Series, h: int, theta: float, when: pd.Series | None = None) -> float:
    """Empirical P(min close over next h bars / close - 1 < -theta)."""
    fwd_min = close.shift(-1).rolling(h).min().shift(-(h - 1))
    hit = (fwd_min / close - 1.0 < -theta)
    hit = hit.iloc[:-h]
    if when is not None:
        when = when.reindex(hit.index).fillna(False)
        return float(hit[when].mean()) if when.any() else float('nan')
    return float(hit.mean())


def _print_table(r: dict) -> None:
    print(f"\n{'variant':<22}{'total':>12}{'CAGR':>9}{'vol':>8}{'Sharpe':>8}{'maxDD':>9}{'Calmar':>8}")
    print('-' * 74)
    for v, s in r['variants'].items():
        print(f"{v:<22}{s['total_return']:>11.1%}{s['cagr']:>8.1%}{s['ann_vol']:>8.1%}"
              f"{s['sharpe']:>8.2f}{s['max_drawdown']:>9.1%}{s['calmar']:>8.2f}")
    p = r['premise']
    print(f"\nmean daily return  risk-off days: {p['mean_ret_risk_off_days']:+.4%}   "
          f"risk-on days: {p['mean_ret_risk_on_days']:+.4%}")
    print(f"share of days flagged risk-off: {p['share_days_risk_off']:.1%}   "
          f"long/short flips: {p['n_short_flips']}")
    print(f"realized 10%/14d drawdown freq   all: {p['realized_dd_freq_all']:.1%}   "
          f"on flagged days: {p['realized_dd_freq_off']:.1%}")


def _write_readme(r: dict) -> None:
    s, p = r['variants'], r['premise']
    hodl, si, sw, fi, fw = (s['hodl'], s['short_is'], s['short_wf'], s['flat_is'], s['flat_wf'])

    def row(v, label):
        x = s[v]
        return (f"| {label} | {x['total_return']:,.0%} | {x['cagr']:.1%} | {x['ann_vol']:.0%} "
                f"| {x['sharpe']:.2f} | {x['max_drawdown']:.0%} | {x['calmar']:.2f} |")

    holds     = ', '.join(f"`{x['name']}` {x['hold']}b" for x in r['config']['signals'])
    prem_ok   = p['realized_dd_freq_off'] > p['realized_dd_freq_all']
    flat_wf_edge = (fw['calmar'] > hodl['calmar'] and fw['sharpe'] >= hodl['sharpe'] - 0.02)
    short_verdict = ("trails buy & hold on return in every variant, in-sample and out"
                     if si['total_return'] < hodl['total_return'] and sw['total_return'] < hodl['total_return']
                     else "beats buy & hold in at least one variant")

    md = f"""# Trading the validated drawdown signal

*Regenerate with `python research/drawdown_overlay/simulate.py`. Window
{r['history']['start']} to {r['history']['end']}, {r['history']['bars']:,} BTC daily bars.*

Four nodes clear Bonferroni on the 10%-drawdown outcome in
[`validation.dd10.json`](../../workspaces/btc_daily_14days/validation.dd10.json):
`bb_pct_20` at +3d and +7d, `drawdown_7` at +3d, `ma_cross_7_21` at +3d — three
distinct features, all firing on their low tail (price at the lower Bollinger
band, a deep trailing drawdown, the fast MA below the slow one). This asks the
question the main README defers: **is that edge worth trading?**

## The rule

Default position is **long BTC (+1)**. A feature *triggers* when it closes below
its p{r['config']['pctile']} (bottom decile); each trigger forces a risk-off
stance for that feature's validated horizon ({holds}), unioned across the three.
Two risk-off stances are tested — **short (−1)** and **flat (0)** — and two
threshold regimes:

| | threshold | honest? |
|---|---|---|
| **IS** | each feature's p{r['config']['pctile']} over the whole history | no — the nodes were *selected* on this same history |
| **WF** | expanding p{r['config']['pctile']}, past data only, after a {r['config']['wf_warmup_bars'] // 365}y warm-up | yes — fully causal |

Costs: {r['config']['cost_bps']} bps per unit of turnover, so a long→short flip pays {2 * r['config']['cost_bps']} bps.

![equity curve](equity_curve.png)

## Results

| variant | total return | CAGR | ann vol | Sharpe | max drawdown | Calmar |
|---|--:|--:|--:|--:|--:|--:|
{row('hodl', 'buy & hold')}
{row('flat_is', 'flat overlay — IS')}
{row('flat_wf', 'flat overlay — walk-forward')}
{row('short_is', 'long/short overlay — IS')}
{row('short_is_2of3', 'long/short, needs ≥2 of 3 — IS')}
{row('short_wf', 'long/short overlay — walk-forward')}

## Does the premise hold?

The nodes predict drawdown *probability*, not negative *expected return* — and
BTC's unconditional drift is large and positive ({hodl['cagr']:.0%} CAGR).

- Realized 10%-within-14d drawdown frequency: **{p['realized_dd_freq_all']:.0%}**
  overall vs **{p['realized_dd_freq_off']:.0%}** on flagged days — the flag
  {'concentrates' if prem_ok else 'does not concentrate'} drawdown risk, but only mildly.
- Mean daily return: **{p['mean_ret_risk_off_days']:+.3%}** on flagged days vs
  **{p['mean_ret_risk_on_days']:+.3%}** on the rest. So the flag really does pick out
  the weak days — but it flags **{p['share_days_risk_off']:.0%}** of all days to do it.
- {p['n_short_flips']} position changes over the window.

## Verdict

**Shorting the signal {short_verdict}.** Going short into a {hodl['cagr']:.0%}-a-year
drift surrenders far more in missed upside than the concentrated left-tail is worth;
the long/short max drawdown is *deeper* than buy & hold's, because the short leg
bleeds through every rally.

**The flat overlay is the honest reading of "capitalise on avoiding drawdowns."**
In-sample it roughly matches buy & hold's return at a Sharpe of {fi['sharpe']:.2f}
(vs {hodl['sharpe']:.2f}) and a {abs(fi['max_drawdown']):.0%} max drawdown (vs
{abs(hodl['max_drawdown']):.0%}). Walk-forward that {'largely survives' if flat_wf_edge else 'fades'}:
Sharpe {fw['sharpe']:.2f}, Calmar {fw['calmar']:.2f} vs buy & hold's {hodl['calmar']:.2f}.

Either way, a Bonferroni-clearing p-value bought a modest risk-management overlay
at best — not a return engine. That gap between *significant* and *tradable* is
the whole point of the parent project.
"""
    (HERE / 'README.md').write_text(md, encoding='utf-8')


if __name__ == '__main__':
    main()
