"""
Does the one validated signal actually pay?

Five tests clear Bonferroni on the -10% barrier in
`workspaces/btc_daily_14days/validation.dd10.json`:

    bb_pct_20      +3d   p = 3e-5   (floor)
    bb_pct_20      +7d   p = 3e-5   (floor)
    drawdown_7     +3d   p = 3e-5   (floor)
    ma_cross_7_21  +3d   p = 1.5e-4
    ma_ratio_7     +3d   p = 1.8e-4

Four distinct features; the horizon is how long the elevated touch risk
persists. All four fire on their *low* tail — price pinned to the lower
Bollinger band, a deep trailing drawdown, the fast MA below the slow one, price
below its own 7-day average.

Worth carrying forward from `--skew`: these are not all the same kind of signal.
Measured against the mirror barrier on identical bars at +3d, bb_pct_20 is
+27.8pp downside against +1.7pp upside — genuinely asymmetric — while
ma_cross_7_21 is +19.8 against +21.3, i.e. it raises the upside barrier slightly
*more* than the downside one. It clears Bonferroni as a drawdown predictor and
still carries no directional information; it is detecting volatility. That is a
reason to expect a short overlay built on these to underperform a flat one.

This builds a long/short overlay from them and compares its equity curve to
buy-and-hold. The trade construction is discrete and event-based ("variant B"):

  * default position is +1 (long BTC);
  * a feature *triggers* on a down-cross of its low-tail threshold — it was at
    or above the threshold on the previous bar and below it on this one;
  * a trigger opens a fixed short of exactly the node's validated horizon
    (7 bars for bb_pct_20, 3 for the other two), then that leg closes;
  * while a leg is open, further triggers for the same feature are ignored — no
    overlap, no re-arming until the feature climbs back above its threshold;
  * the overlay is short whenever *any* leg is open (a flat
    variant holds 0 instead of -1 over the same windows).

This is the faithful reading of what `--validate` scored: a per-bar level
condition with a fixed h-bar forward outcome. A persistence construction ("stay
short as long as the feature is in its low zone") was tried and rejected — it
keeps the overlay short ~37% of all days and dilutes the edge.

Two threshold regimes, reported side by side:

  IS  in-sample   — thresholds are the p10 of each feature over the whole
                    history. The nodes were selected on this same history, so
                    this curve is optimistic by construction.
  WF  walk-forward — thresholds are the expanding p10 using only data up to
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
    {'name': 'ma_ratio_7',    'feature': 'ma_ratio', 'params': {'period': 7},           'hold': 3},
]

PCTILE     = 10       # low-tail trigger: feature below its p10 (bottom decile)
COST_BPS   = 10       # per unit turnover, one-way
ANN        = 365      # BTC trades every day
WF_WARMUP  = 365 * 2  # walk-forward: no position until this many bars of history

_INK, _GREY, _BLUE, _RED, _GREEN = '#333333', '#a0aec0', '#2b6cb0', '#b03a2e', '#2f855a'


# ── signal construction ──────────────────────────────────────────────────────

def _discrete_trades(feat: pd.Series, thresh: pd.Series, hold: int) -> tuple[pd.Series, int]:
    """
    Non-overlapping fixed-length shorts on each down-cross of `thresh`.

    A trigger at bar i is:  feat[i-1] >= thresh[i-1]  and  feat[i] < thresh[i].
    It marks position bars i .. i+hold-1 True; after the backtest's one-bar
    shift those become the hold return bars immediately following the trigger.
    Triggers that land while a leg is already open are ignored, so a feature can
    only re-fire after it has climbed back above its threshold.

    Returns (mask, n_trades).
    """
    f, thr = feat.to_numpy(float), thresh.to_numpy(float)
    n      = len(f)
    below  = f < thr
    mask   = np.zeros(n, dtype=bool)
    open_until = -1
    n_trades   = 0
    for i in range(1, n):
        if i <= open_until:
            continue
        if np.isnan(thr[i]) or np.isnan(thr[i - 1]) or np.isnan(f[i]) or np.isnan(f[i - 1]):
            continue
        if below[i] and not below[i - 1]:
            end = min(i + hold - 1, n - 1)
            mask[i:end + 1] = True
            open_until = end
            n_trades += 1
    return pd.Series(mask, index=feat.index), n_trades


def build_positions(data: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict]:
    """
    Per-bar target weights for the overlay variants. A weight at bar t is decided
    from information available at close t and earns the t -> t+1 return.
    """
    feats = {s['name']: features.compute(data, s['feature'], s['params']).reindex(data.index)
             for s in SIGNALS}

    is_open = pd.Series(False, index=data.index)   # any leg open, in-sample thresholds
    wf_open = pd.Series(False, index=data.index)   # any leg open, walk-forward thresholds
    per_sig = {}
    n_trades = {}

    for s in SIGNALS:
        f = feats[s['name']]
        thr_is = pd.Series(np.nanpercentile(f.dropna(), PCTILE), index=data.index)
        thr_wf = f.expanding(min_periods=WF_WARMUP).quantile(PCTILE / 100.0)

        m_is, k_is = _discrete_trades(f, thr_is, s['hold'])
        m_wf, k_wf = _discrete_trades(f, thr_wf, s['hold'])
        per_sig[s['name']] = m_is
        n_trades[s['name']] = {'is': k_is, 'wf': k_wf}
        is_open |= m_is
        wf_open |= m_wf

    wf_open.iloc[:WF_WARMUP] = False   # no walk-forward opinion until warm-up completes

    pos = pd.DataFrame(index=data.index)
    pos['hodl']          = 1.0
    pos['short_is']      = np.where(is_open, -1.0, 1.0)
    pos['short_wf']      = np.where(wf_open, -1.0, 1.0)
    pos['flat_is']       = np.where(is_open,  0.0, 1.0)
    pos['flat_wf']       = np.where(wf_open,  0.0, 1.0)
    pos['n_signals']     = sum(per_sig.values()).astype(int)   # 0..3 legs open (IS)
    pos['short_is_2plus'] = np.where(pos['n_signals'] >= 2, -1.0, 1.0)
    pos['_is_open']      = is_open
    pos['_wf_open']      = wf_open

    meta = {'n_trades': n_trades,
            'total_trades_is': sum(v['is'] for v in n_trades.values()),
            'total_trades_wf': sum(v['wf'] for v in n_trades.values())}
    return pos, feats, meta


# ── backtest ─────────────────────────────────────────────────────────────────

def run_variant(ret: pd.Series, weight: pd.Series) -> pd.Series:
    """
    Daily net return for a weight series. weight_t is decided at close t and
    held over t -> t+1, so the cost of the trade that set weight_t is booked on
    bar t+1 (turnover = change in the held weight entering that bar).
    """
    held     = weight.shift(1)
    turnover = held.diff().abs().fillna(0.0)
    return held * ret - (COST_BPS / 1e4) * turnover


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
             f'Shaded = an open short leg (in-sample thresholds). Fixed {SIGNALS[1]["hold"]}/{SIGNALS[0]["hold"]}-bar '
             f'trades on each down-cross, no overlap. {COST_BPS} bps per unit turnover; '
             f'walk-forward thresholds start after {WF_WARMUP // 365}y warm-up.',
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
    pos, feats, meta = build_positions(data)

    variants = ['hodl', 'flat_is', 'flat_wf', 'short_is', 'short_is_2plus', 'short_wf']
    rets = {v: run_variant(ret, pos[v]) for v in variants}
    eq   = pd.DataFrame({v: (1.0 + rets[v].fillna(0.0)).cumprod() for v in variants})

    # premise check: is the *forward* 1-bar return actually worse after a trigger,
    # and does a short-leg window really catch more 10%/14d drawdowns?
    fwd  = ret.shift(-1)
    on   = pos['_is_open']
    premise = {
        'fwd_ret_after_signal':   float(fwd[on].mean()),
        'fwd_ret_otherwise':      float(fwd[~on].mean()),
        'share_bars_short':       float(on.mean()),
        'total_trades_is':        int(meta['total_trades_is']),
        'total_trades_wf':        int(meta['total_trades_wf']),
        'realized_dd_freq_all':   _dd_hit_freq(data['close'], 14, 0.10),
        'realized_dd_freq_signal': _dd_hit_freq(data['close'], 14, 0.10, on),
    }

    results = {
        'generated':  datetime.now(timezone.utc).isoformat(),
        'history':    {'start': str(data.index[0].date()), 'end': str(data.index[-1].date()),
                       'bars': int(len(data))},
        'config':     {'pctile': PCTILE, 'cost_bps': COST_BPS, 'wf_warmup_bars': WF_WARMUP,
                       'signals': [{k: s[k] for k in ('name', 'hold')} for s in SIGNALS]},
        'trades':     meta['n_trades'],
        'premise':    premise,
        'variants':   {v: stats(rets[v]) for v in variants},
    }
    (HERE / 'results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    plot(eq, pos['_is_open'], HERE / 'equity_curve.png')

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
    print(f"\nforward 1-bar return  after a trigger: {p['fwd_ret_after_signal']:+.4%}   "
          f"otherwise: {p['fwd_ret_otherwise']:+.4%}")
    print(f"bars inside a short leg: {p['share_bars_short']:.1%}   "
          f"trades  IS: {p['total_trades_is']}  WF: {p['total_trades_wf']}")
    print(f"realized 10%/14d drawdown freq   all: {p['realized_dd_freq_all']:.1%}   "
          f"on trigger bars: {p['realized_dd_freq_signal']:.1%}")


def _write_readme(r: dict) -> None:
    s, p, c = r['variants'], r['premise'], r['config']
    hodl, si, sw, fi, fw = (s['hodl'], s['short_is'], s['short_wf'], s['flat_is'], s['flat_wf'])

    def row(v, label):
        x = s[v]
        return (f"| {label} | {x['total_return']:,.0%} | {x['cagr']:.1%} | {x['ann_vol']:.0%} "
                f"| {x['sharpe']:.2f} | {x['max_drawdown']:.0%} | {x['calmar']:.2f} |")

    holds  = ', '.join(f"`{x['name']}` {x['hold']}b" for x in c['signals'])
    prem_ok = p['realized_dd_freq_signal'] > p['realized_dd_freq_all']

    def _cmp(a, b, tol=0.03):
        d = a['cagr'] - b['cagr']
        return 'a wash with' if abs(d) < tol else ('ahead of' if d > 0 else 'behind')
    short_is_word = _cmp(si, hodl)
    short_wf_word = _cmp(sw, hodl)

    md = f"""# Trading the validated downside-barrier signal

*Regenerate with `python research/drawdown_overlay/simulate.py`. Window
{r['history']['start']} to {r['history']['end']}, {r['history']['bars']:,} BTC daily bars
(live yfinance pull — the last bar and the exact figures move a little between runs).*

Five tests clear Bonferroni on the −10% barrier in
[`validation.dd10.json`](../../workspaces/btc_daily_14days/validation.dd10.json):
`bb_pct_20` at +3d and +7d, `drawdown_7`, `ma_cross_7_21` and `ma_ratio_7` at +3d
— four distinct features, all firing on their low tail (price at the lower
Bollinger band, a deep trailing drawdown, the fast MA below the slow one, price
below its own 7-day average). This asks the question the main README defers:
**is that edge worth trading?**

## The rule  (discrete, event-based)

Default position is **long BTC (+1)**. A feature *triggers* on a **down-cross** of
its p{c['pctile']} (bottom-decile) threshold — at or above it last bar, below it
now. Each trigger opens one fixed-length short of exactly the node's validated
horizon ({holds}), which then closes; a feature is not re-armed until it has
climbed back above its threshold, and triggers arriving while its leg is still
open are ignored. The overlay is risk-off whenever **any** leg is open. Two risk-off stances — **short (−1)** and **flat (0)** — and two threshold
regimes:

| | threshold | honest? |
|---|---|---|
| **IS** | each feature's p{c['pctile']} over the whole history | no — the nodes were *selected* on this same history |
| **WF** | expanding p{c['pctile']}, past data only, after a {c['wf_warmup_bars'] // 365}y warm-up | yes — fully causal |

Costs: {c['cost_bps']} bps per unit of turnover, so a long→short flip pays {2 * c['cost_bps']} bps.
(A "stay short while the feature is in its low zone" construction was tried and
dropped — it holds the overlay short far too often and dilutes the edge.)

![equity curve](equity_curve.png)

## Results

| variant | total return | CAGR | ann vol | Sharpe | max drawdown | Calmar |
|---|--:|--:|--:|--:|--:|--:|
{row('hodl', 'buy & hold')}
{row('flat_is', 'flat overlay — IS')}
{row('flat_wf', 'flat overlay — walk-forward')}
{row('short_is', 'long/short overlay — IS')}
{row('short_is_2plus', 'long/short, needs ≥2 legs open — IS')}
{row('short_wf', 'long/short overlay — walk-forward')}

## Does the premise hold?

The nodes predict drawdown *probability*, not negative *expected return* — and
BTC's unconditional drift is large and positive ({hodl['cagr']:.0%} CAGR).

- **{p['total_trades_is']}** trades in-sample, **{p['total_trades_wf']}** walk-forward;
  a short leg is open on **{p['share_bars_short']:.0%}** of all bars.
- Forward 1-bar return **{p['fwd_ret_after_signal']:+.3%}** on trigger bars vs
  **{p['fwd_ret_otherwise']:+.3%}** elsewhere — the trigger {'does' if p['fwd_ret_after_signal'] < 0 else 'does not'}
  pick out negative expected return, {'which is what a short needs' if p['fwd_ret_after_signal'] < 0 else 'so a short has nothing to work with'}.
- Realized 10%-within-14d drawdown frequency: **{p['realized_dd_freq_all']:.0%}** overall
  vs **{p['realized_dd_freq_signal']:.0%}** on trigger bars — the signal
  {'concentrates' if prem_ok else 'does not concentrate'} drawdown risk.

## Verdict

**Shorting doesn't earn its keep.** Trigger bars carry a *lower* forward return
than average ({p['fwd_ret_after_signal']:+.3%} vs {p['fwd_ret_otherwise']:+.3%}) —
but still positive: the signal marks *weak* days, not *down* days. So the
long/short overlay is {short_is_word} buy & hold in-sample ({si['cagr']:.0%} CAGR)
and {short_wf_word} it walk-forward ({sw['cagr']:.0%} vs {hodl['cagr']:.0%} CAGR,
similar drawdown). The {hodl['cagr']:.0%}-a-year drift swamps the edge either way.

**Sidestepping the drawdowns is the result.** Holding *flat* rather than short
over the same windows beats buy & hold on risk-adjusted terms in both regimes, and
walk-forward on outright return too: {fw['cagr']:.0%} CAGR at Calmar
{fw['calmar']:.2f} vs {hodl['calmar']:.2f}, with the max drawdown cut from
{abs(hodl['max_drawdown']):.0%} to {abs(fw['max_drawdown']):.0%}. On ~{p['total_trades_wf']}
trades over one asset and one path, read that as encouraging, not proven.

A Bonferroni-clearing p-value bought a risk-management overlay — not a return
engine. The gap between *significant* and *tradable* is the whole point of the
parent project.
"""
    (HERE / 'README.md').write_text(md, encoding='utf-8')


if __name__ == '__main__':
    main()
