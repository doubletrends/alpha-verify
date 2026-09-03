![Python](https://img.shields.io/badge/Python-v3.12-3776AB?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-v2.0-013243?logo=numpy&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-v2.2-150458?logo=pandas&logoColor=white)
![openpyxl](https://img.shields.io/badge/openpyxl-v3.1-1D6F42)

# Where Will Price Reach? — Conditional Barrier Probabilities on Any Asset

> "Where will price *be* on Friday" is the wrong question. A stop, a limit, a
> liquidation and a margin call all respond to something else: **did price ever reach
> this level?** This measures that — every barrier level, every condition, on intraday
> extremes, against a shuffled null.

Six artifacts per node, in order:

```
cubes_full/<family>/<node>.npz        measure   41 θ × 10 bins × 30 h
surfaces_full/<family>/<node>.xlsx    render    the same values, readable
cubes_summary/<family>/<node>.npz     reduce    17 θ × 10 bins × 7 h
surfaces_summary/<family>/<node>.xlsx render
validations/<family>/<node>.npz       falsify   exact shuffled null
validations_xlsx/<family>/<node>.xlsx render    readable validation surfaces
```

**The full grid is the faithful record and is never judged. The summary is what gets
judged.** The summary is a strict *subset* — index selection out of the full cube, never
interpolation, so a summary cell is bit-identical to the full cell it came from.

That split exists because adjacent cells of the full grid are very nearly the same
measurement. A peak taken over 12,300 cells carries a large noise ceiling by
construction, and counting those cells as 12,300 independent tests overstates the search
in both directions at once. On the summary the same statistic is a peak over 1,190
cells, with a correspondingly lower ceiling and an honest test count.

Every node passes through every stage, the `baseline` node included — its validation
comes out degenerate by construction (peak deviation exactly 0, p exactly 1), and that
is on purpose. A uniform pipeline carrying one meaningless file is worth more than a
pipeline with a special case in it.

The **cube** is a 3-D array per node — barrier level θ from −20% to +20% (41 levels,
including 0), the feature's ten condition bins, and every horizon from +1 to +30 bars.
Each entry is the raw conditional probability:

$$
P\left(\text{price touches } \θ \text{ within } h \;\middle|\; X_t \in \text{bin}\right)
$$

Nothing is subtracted. A cell reads on its own terms — *under this condition a −5% touch
within 7 days happens 50% of the time* — with no base rate to carry in your head.

**The base rate is a node.** `baseline` has a constant feature, so every bar falls in one
bin and its single column *is* P(touch θ in h) unconditional. It flows through `--cubes`
and `--surfaces` like any other node, with no special case anywhere in the engine, and
the filters read it from there. It is built first, since it is the reference everything
else is judged against.

```
  unconditional P(touch θ in h) — BTC daily, from cubes/_base/baseline.npz

   θ      +1d     +3d     +7d    +14d    +30d
    +20%     0.2%    1.5%    6.8%   16.3%   31.6%
    +10%     2.3%    9.6%   22.5%   37.4%   56.2%
     +5%    10.3%   28.1%   46.9%   62.2%   75.8%
      0%    97.9%   98.8%   99.2%   99.3%   99.6%
     -5%    10.2%   24.6%   39.5%   51.2%   62.3%
    -10%     2.3%    8.4%   17.6%   28.7%   41.0%
    -20%     0.2%    1.3%    4.8%    9.7%   19.7%
```

BTC's upward drift is right there: +10% within 30 days is reached 56.2% of the time
against 41.0% for −10%. Any conditional surface has to beat *that*, not a coin flip.

One caveat the design inherits: the baseline is measured over every bar, while a node
with a long warmup — a 200-bar moving average — is measured over fewer, and the bars it
drops are the oldest and most volatile. For those features the comparison is very
slightly against a different history.

About 90 KB per node, and it carries the event counts too, so a follow-up question needs
no recomputation.

The full horizon ladder is where the *shape* of a signal lives. `bb_pct_20`'s bottom
decile at θ = −5% runs +14.4 pp at h = 3, decays to +7.6 pp by h = 22, then curls back
up — a profile no set of four sheets would reveal.

The **workbook** is a faithful rendering of that cube, not a summary of it: **one tab
per condition bin**, and each tab is that bin's entire θ × h face. Ten tabs × 41 θ × 30
horizons is exactly the 12,300 values the cube holds — nothing is dropped.

Rows run the way a price ladder reads, highest barrier at the top:

```
  bb_pct_20 — condition: bb_pct in x < 0.1213          (tab 1 of 10)

  θ    +1d    +2d    +3d    +5d    +7d   +10d   +14d   +21d   +30d
  n =      425    425    425    425    425    425    425    425    425
   +20%   0.0%   1.2%   2.1%   3.5%   7.3%  10.4%  12.9%  20.0%  26.1%
   +10%   5.2%   8.7%  13.9%  19.8%  24.7%  29.2%  36.0%  43.1%  49.7%
    +5%  14.6%  26.1%  33.2%  45.2%  52.2%  60.2%  66.1%  71.1%  73.7%
     0%  97.7%  98.6%  98.6%  99.3%  99.5%  99.5%  99.5%  99.8%  99.8%
    -5%  22.6%  32.7%  38.8%  46.1%  50.1%  57.9%  60.9%  65.2%  72.9%
   -10%   6.4%  12.5%  17.2%  23.8%  27.1%  31.5%  39.5%  46.8%  52.5%
   -20%   0.5%   2.4%   4.9%   8.9%  11.1%  13.4%  16.0%  21.9%  27.8%
```

*(BTC daily, bottom decile — price pinned to the lower Bollinger band.)*

Read it as a band that widens with horizon. The asymmetry is right there without any
arithmetic: at +1d the **−5%** row is 22.6% against **+5%** at 14.6% — same condition,
same bars, so this condition reaches down more readily than up. Flip to tab 10 and the
band tilts the other way.

θ = 0 sits shaded in the middle. It is near-degenerate by construction — it asks whether
the high ever returned to the entry close — and it marks the boundary between the two
questions: above it a cell asks whether the *high* reached that level, below it whether
the *low* did.

The colour scale is a fixed 0–100% on every tab, so flipping between them shows the band
moving with the condition rather than each tab being rescaled to look alike.

* * *

## What makes the numbers trustworthy

**Intraday extremes, not closes.** A barrier is touched when the bar's low or high
reaches it. Every close-to-close measurement understates this badly:

| barrier | on closes | on high/low | |
|---|--:|--:|--:|
| −10% within 14d | 21.7% | **28.7%** | +7.0 pp |
| −5% within 7d | 28.6% | **39.5%** | +10.8 pp |
| −5% within 3d | 15.4% | **24.6%** | +9.2 pp |

A stop that only fires on closes is not a stop. Everything here uses `low`/`high`, and
the window opens at *t+1* — a barrier cannot be touched on the bar the condition is read
on.

**Quantile bins, not equal-width slices.** Conditions are the feature's deciles, so every
column has a known, equal sample behind it and each one is a condition you could state
out loud ("the feature is in its bottom tenth"). Equal-width bins put almost nothing in
the tails, which is exactly where the interesting conditions live.

**Two gates, both required.** A cube has 41 barrier levels × 10 bins × 30 horizons. The largest of 400 noisy estimates is well into the tens of pp even on a feature
carrying nothing, so neither filter alone means anything:

- **Economic** — a cell deviates from its base rate by ≥ `min_dev` pp, in a bin with
  ≥ `min_bin_n` observations, across ≥ `min_run` **adjacent θ rows** with the same sign.
  The adjacency is the point: an edge at exactly −7% and nowhere else is not a stop
  level, it is an artifact. A *band* of barrier levels moving together is both harder to
  produce by accident and the only shape you could act on.
- **Statistical** — the surface's peak clears a shuffled null, corrected across the
  sweep. Shifting the feature (rather than the outcome) keeps the two forward-excursion
  series consistent with each other and with the price path, preserves the feature's
  autocorrelation, and destroys only its alignment with the future.

### The null is exact, and its floor is real

Writing `counts[θ, bin]` as a function of shift makes it a circular cross-correlation,
so **one FFT yields every shift at once** — about 70× faster than resampling, and it
returns the whole permutation distribution instead of a sample from it. All 30 horizons
for a workspace take ~3 minutes.

That exactness exposes something resampling hid. There are only *n* distinct circular
shifts, so the finest obtainable p-value is `1/(n+1)` — about 2.6 × 10⁻⁴ on eleven years
of daily bars. Drawing 20,000 random shifts from a group of 4,214 and reporting
`p = 1/20001` claims a resolution the data cannot produce, overstating significance by
roughly 5×.

Because that floor sits **above** the Bonferroni threshold for a sweep this size, family-
wise correction is structurally impossible here — nothing could ever clear it, however
strong the signal:

```
p-value floor                       2.59e-04
Bonferroni α over 462 tests         1.08e-04   below the floor, unusable
```

Reducing to the summary grid shrinks the sweep from 1,980 tests to 462 and moves
Bonferroni's threshold from 2.5e-05 to 1.1e-04 — much closer to reachable, still not
reachable.

So the gate uses **Benjamini–Hochberg** instead, comparing the k-th smallest p-value
against `k·q/m`. Tests sitting at the floor then clear collectively, and what is
controlled is the share of false positives among discoveries — the right target for a
screen of this size.

Verdicts are assigned at `--gate`, never stored per node: a multiple-testing correction
is a property of the collection, and a verdict baked into a node's artifact would go
stale the moment another node joined the sweep.

* * *

## Results

| Workspace | Nodes | Tests | Economic | BH discovery | **Cleared both** |
|---|--:|--:|--:|--:|--:|
| `btc_daily_14days` | 66 | 462 | 54 | 51 | **50** |
| `nasdaq_hourly_24hrs` | 34 | 204 | 31 | 30 | **29** |

Barrier-touch probability is strongly and widely predictable, which is not a surprise —
many discoveries sit at the resolution floor, meaning the real surface is more extreme
than *every* one of the ~3,838 usable shifts.

**Read the horizon distribution before getting excited.** Of the 50 cleared BTC nodes,
36 have their strongest cell at **h = 1**. At one bar ahead, "will price touch −2%" is
close to asking "is volatility high right now", and volatility features answer that
near-tautologically. It is real, it is significant, and it is mostly mechanical.

`cycle` and `dxy` clear nothing, on either grid, in every version of this pipeline.

**Barrier-touch probability is strongly predictable, and that is not a surprise.** What
drives it is conditional *variance* — the size of the coming move — which is what
volatility clustering has always said. Sorted by which families clear, the picture is
plain: `ma`, `volatility`, `roc`, `macd` and `drawdown` clear almost everywhere;
`cycle`, `dxy` and `on_chain` clear **nowhere on BTC**, and `treasury` clears nowhere on
NASDAQ. The halving cycle and MVRV — two of the most-cited Bitcoin indicators — produce
not one significant cell across 40 barrier levels, 4 horizons and 10 conditions each.

The scan finds conditional variance, not conditional mean. That distinction is the whole
value: it tells you where to put a stop, and it does not tell you which way to bet.

* * *

## Use it

```bash
pip install -r requirements.txt

python run.py --workspace btc_daily_14days --cubes              # 1. measure  (full)
python run.py --workspace btc_daily_14days --surfaces           # 2. render   (full)
python run.py --workspace btc_daily_14days --cubes-summary      # 3. reduce   (+ economic filter)
python run.py --workspace btc_daily_14days --surfaces-summary   # 4. render   (summary)
python run.py --workspace btc_daily_14days --validate           # 5. falsify  (on the summary)
python run.py --workspace btc_daily_14days --gate               # 6. correct + intersect
python run.py --workspace btc_daily_14days --status             # the funnel
python run.py --workspace btc_daily_14days --read bb_pct_20     # one node, in the terminal
```

| Command | Writes |
|---|---|
| `--cubes` | `cubes/<family>/<node>.npz` (**the measurement**), `evaluation.json` — builds `baseline` first |
| `--surfaces` | `surfaces_full/<family>/<node>.xlsx` — 10 tabs, renders only |
| `--cubes-summary` | `cubes_summary/…` + `evaluation.json` — the judged grid |
| `--surfaces-summary` | `surfaces_summary/…` — same renderer, coarse grid |
| `--validate` | `validations/<family>/<node>.npz` — per-cell and peak nulls, one per node |
| `--validate` | `validations_xlsx/<family>/<node>.xlsx` — readable validation summary, signed deviations and pointwise p-values |
| `--gate` | `cleared.json` — BH across the sweep, intersected with the economic filter |
| `--status` | *(terminal)* the funnel per family |
| `--read <id>` | *(terminal)* the θ rows carrying a qualifying cell |

`--family <name>` restricts any build; `--rerun` rebuilds existing cubes;
`--fdr Q` sets the Benjamini-Hochberg rate at `--gate` (default 0.05).

* * *

## Add your own asset

A workspace is one `universe.json`. No root code changes.

```json
{
  "meta": {
    "asset": {"provider": "yfinance", "ticker": "BTC-USD", "interval": "1d"},
    "start_date": "2015-01-01",
    "horizons": {"min": 1, "max": 30},
    "barrier_horizons": [3, 7, 14, 30],
    "theta":    {"min": -0.20, "max": 0.20, "step": 0.01},
    "summary":  {"theta_abs": [0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20],
                 "horizons":  [1, 2, 3, 5, 7, 14, 30]},
    "n_bins":   10,
    "evaluate": {"min_dev": 10.0, "min_bin_n": 50, "min_run": 2}
  },
  "families": {
    "rsi": [
      {"id": "rsi_14", "family": "rsi", "category": "price_momentum",
       "feature": "rsi", "params": {"period": 14}, "data": ["ohlcv"]}
    ]
  }
}
```

`summary.theta_abs` is given as magnitudes and mirrored, so the judged ladder stays
symmetric and 0 appears exactly once. Every value must exist on the full grid — the
summary is a subset, and asking for one that isn't there raises rather than
interpolating, because interpolating would make it a second measurement.

**Scale θ to the asset and horizon.** BTC daily runs ±20% in 1% steps. NASDAQ hourly runs
±4% in 0.2% steps — a 20% move inside a trading day does not happen, and every row of
that ladder would be empty. Getting this wrong is the one setup mistake that silently
produces a sheet full of zeros.

A **cross-asset feature** goes in `data/features.py`:

```python
def _my_feature(close: pd.Series, period: int) -> pd.Series: ...

register('my_feature', lambda d, p: _my_feature(d['close'], p['period']))
```

A **workspace-specific feature or data source** goes in `workspaces/<name>/plugin.py`,
imported automatically when the workspace loads:

```python
from data import features, fetcher

def _my_source(start: str, asset: dict) -> pd.DataFrame:   # DatetimeIndex-ed frame
    ...

fetcher.register_source('my_source', _my_source)
```

`universe.json` is a pure declaration and is never written back — a node's progress is
read off the filesystem, so a partial run resumes by looking at what exists.

* * *

## Layout

```
run.py               ← CLI: --surfaces / --validate / --gate / --status / --read
workspace.py         ← Workspace: config, paths, plugin loading
data/
  features.py        ← FeatureRegistry: register() / compute()
  fetcher.py         ← SourceRegistry: register_source() / fetch()
engine/
  barrier.py         ← the cube, quantile bins, summary selection, economic filter, npz io
  validate.py        ← exact FFT circular-shift null, Benjamini-Hochberg
  writer.py          ← the xlsx deliverable (rendering only)
tree/tree.py         ← node traversal over universe.json
workspaces/<name>/
  universe.json      ← nodes + all asset config
  plugin.py          ← (optional) custom features and sources
  cubes/_base/baseline.npz         ← the unconditional reference, an ordinary node
  cubes/<family>/<node>.npz        ← the measurement (θ x bin x horizon)
  surfaces/<family>/<node>.xlsx    ← the deliverable
  evaluation.json  validation.json  cleared.json
```

### Built-in data sources

| Key | Provider | Columns |
|---|---|---|
| `ohlcv` | yfinance (workspace asset + interval) | open, high, low, close, volume |
| `vix` | yfinance `^VIX` | vix |
| `treasury` | yfinance `^TNX` | tnx |
| `dxy` | yfinance `DX-Y.NYB` (daily) | dxy |
| `coinmetrics` | CoinMetrics community API *(btc plugin)* | mvrv, hash_rate, adr_act_cnt, tx_cnt |

* * *

## Two limits worth knowing

**Sample size.** A decile condition on ten years of daily bars is ~425 bars, but only
~100–200 *non-overlapping* windows. Every count in the sheet is a count of bars; treat
the independent sample as far smaller than it looks, particularly at long horizons.

**Bar resolution.** Daily OHLC records the high and the low of a session but not their
order. Any question that depends on which came first — and that includes anything with
both a take-profit and a stop inside a few percent — is not answerable from daily bars
at all. That is a data problem, not a code one.

* * *

## References

**Data** — [`yfinance`](https://github.com/ranaroussi/yfinance) ·
[CoinMetrics Community API](https://docs.coinmetrics.io/api/v4)

**Method**
- Davison & Hinkley (1997), *Bootstrap Methods and Their Application*, CUP §4.2 — add-one permutation p-values and resampling resolution
- Benjamini & Hochberg (1995), *Controlling the False Discovery Rate*, JRSS-B 57(1) — the correction the gate applies
- Politis & Romano (1994), *The Stationary Bootstrap*, JASA 89(428) — resampling that preserves serial dependence
- White (2000), *A Reality Check for Data Snooping*, Econometrica 68(5) — multiple testing over a searched universe of rules
- López de Prado (2018), *Advances in Financial Machine Learning*, ch. 3 — barrier labelling of financial time series
- Karatzas & Shreve (1991), *Brownian Motion and Stochastic Calculus* §3.5 — first-passage probabilities

**Indicators** — Wilder (1978) RSI/ATR · Appel (2005) MACD · Bollinger (2001) Bollinger Bands
