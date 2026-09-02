# Trading the validated drawdown signal

*Regenerate with `python research/drawdown_overlay/simulate.py`. Window
2015-01-01 to 2026-09-02, 4,263 BTC daily bars.*

Four nodes clear Bonferroni on the 10%-drawdown outcome in
[`validation.dd10.json`](../../workspaces/btc_daily_14days/validation.dd10.json):
`bb_pct_20` at +3d and +7d, `drawdown_7` at +3d, `ma_cross_7_21` at +3d — three
distinct features, all firing on their low tail (price at the lower Bollinger
band, a deep trailing drawdown, the fast MA below the slow one). This asks the
question the main README defers: **is that edge worth trading?**

## The rule

Default position is **long BTC (+1)**. A feature *triggers* when it closes below
its p10 (bottom decile); each trigger forces a risk-off
stance for that feature's validated horizon (`bb_pct_20` 7b, `drawdown_7` 3b, `ma_cross_7_21` 3b), unioned across the three.
Two risk-off stances are tested — **short (−1)** and **flat (0)** — and two
threshold regimes:

| | threshold | honest? |
|---|---|---|
| **IS** | each feature's p10 over the whole history | no — the nodes were *selected* on this same history |
| **WF** | expanding p10, past data only, after a 2y warm-up | yes — fully causal |

Costs: 10 bps per unit of turnover, so a long→short flip pays 20 bps.

![equity curve](equity_curve.png)

## Results

| variant | total return | CAGR | ann vol | Sharpe | max drawdown | Calmar |
|---|--:|--:|--:|--:|--:|--:|
| buy & hold | 24,204% | 60.1% | 66% | 1.04 | -83% | 0.72 |
| flat overlay — IS | 22,037% | 58.8% | 46% | 1.24 | -65% | 0.91 |
| flat overlay — walk-forward | 29,513% | 62.8% | 49% | 1.24 | -64% | 0.98 |
| long/short overlay — IS | 1,222% | 24.7% | 66% | 0.66 | -85% | 0.29 |
| long/short, needs ≥2 of 3 — IS | 12,606% | 51.4% | 66% | 0.96 | -70% | 0.74 |
| long/short overlay — walk-forward | 3,226% | 35.0% | 66% | 0.78 | -84% | 0.42 |

## Does the premise hold?

The nodes predict drawdown *probability*, not negative *expected return* — and
BTC's unconditional drift is large and positive (60% CAGR).

- Realized 10%-within-14d drawdown frequency: **22%**
  overall vs **25%** on flagged days — the flag
  concentrates drawdown risk, but only mildly.
- Mean daily return: **-0.310%** on flagged days vs
  **+0.478%** on the rest. So the flag really does pick out
  the weak days — but it flags **37%** of all days to do it.
- 228 position changes over the window.

## Verdict

**Shorting the signal trails buy & hold on return in every variant, in-sample and out.** Going short into a 60%-a-year
drift surrenders far more in missed upside than the concentrated left-tail is worth;
the long/short max drawdown is *deeper* than buy & hold's, because the short leg
bleeds through every rally.

**The flat overlay is the honest reading of "capitalise on avoiding drawdowns."**
In-sample it roughly matches buy & hold's return at a Sharpe of 1.24
(vs 1.04) and a 65% max drawdown (vs
83%). Walk-forward that largely survives:
Sharpe 1.24, Calmar 0.98 vs buy & hold's 0.72.

Either way, a Bonferroni-clearing p-value bought a modest risk-management overlay
at best — not a return engine. That gap between *significant* and *tradable* is
the whole point of the parent project.
