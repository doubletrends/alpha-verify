![Python](https://img.shields.io/badge/Python-v3.12-3776AB?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-v2.0-013243?logo=numpy&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-v2.2-150458?logo=pandas&logoColor=white)
![openpyxl](https://img.shields.io/badge/openpyxl-v3.1-1D6F42)

# Where Will Price Reach? — Conditional Barrier Probabilities on Any Asset

> "Where will price *be* on Friday" is the wrong question. A stop, a limit, a
> liquidation and a margin call all respond to something else: **did price ever reach
> this level?** This measures that — every barrier level, every condition, on intraday
> extremes, against a shuffled null.

The pipeline produces exactly one deliverable, per node:

```
surfaces/<family>/<node>.xlsx
```

Four sheets, one per horizon. **Rows** are barrier levels θ from −20% to +20%.
**Columns** are the feature's condition bins. **Cells** are

$$
P\left(\text{price touches } \theta \text{ within } h \mid X_t \in \text{bin}\right)
$$

with the event count beside each one, and the unconditional base rate in column B so
every cell reads against the row it sits on.

```
theta    base  |  x < 0.121  hits |  0.121..0.248 hits |  ...  |  x > 0.967  hits
  n =   4,237  |        425       |          424       |       |        425
 -20%    4.6%  |       11.1%   47 |         3.8%    16 |       |       3.1%   13
 -10%   17.4%  |       27.1%  115 |        18.6%    79 |       |      13.6%   58
  -5%   39.3%  |       50.1%  213 |        41.3%   175 |       |      33.6%  143
  +5%   46.8%  |       52.2%  221 |        44.6%   189 |       |      56.5%  240
 +10%   22.5%  |       24.7%  105 |        20.3%    86 |       |      31.5%  134
 +20%    6.8%  |        7.3%   31 |         5.2%    22 |       |      14.6%   62
```

*(`bb_pct_20`, +7d, BTC daily. Bottom decile — price pinned to the lower Bollinger band —
lifts every downside barrier; the top decile lifts every upside one.)*

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

**Two gates, both required.** A surface has ~40 barrier levels × 10 bins = 400 cells per
horizon. The largest of 400 noisy estimates is well into the tens of pp even on a feature
carrying nothing, so neither filter alone means anything:

- **Economic** — a cell deviates from its base rate by ≥ `min_dev` pp, in a bin with
  ≥ `min_bin_n` observations, across ≥ `min_run` **adjacent θ rows** with the same sign.
  The adjacency is the point: an edge at exactly −7% and nowhere else is not a stop
  level, it is an artifact. A *band* of barrier levels moving together is both harder to
  produce by accident and the only shape you could act on.
- **Statistical** — the surface's peak clears a null built by circularly shifting the
  feature against price, Bonferroni-corrected across the sweep. Shifting the feature
  (rather than the outcome) keeps the two forward-excursion series consistent with each
  other and with the price path, preserves the feature's autocorrelation, and destroys
  only its alignment with the future.

* * *

## Results

| Workspace | Nodes | Surface | Economic | Significant | **Cleared both** |
|---|--:|--:|--:|--:|--:|
| `btc_daily_14days` | 65 | 65 | 56 | 44 | **30** |
| `nasdaq_hourly_24hrs` | 33 | 33 | 30 | 28 | **23** |

Against the null, BTC returns 68 of 240 tests clearing Bonferroni and 81 nominal against
12 expected by chance; NASDAQ returns 72 of 132 clearing and 28 nominal against 6.6.

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

python run.py --workspace btc_daily_14days --surfaces           # build every surface
python run.py --workspace btc_daily_14days --validate           # null-test them
python run.py --workspace btc_daily_14days --gate               # intersect both filters
python run.py --workspace btc_daily_14days --status             # the funnel
python run.py --workspace btc_daily_14days --read bb_pct_20     # one node, in the terminal
```

| Command | Writes |
|---|---|
| `--surfaces` | `surfaces/<family>/<node>.xlsx` (**the deliverable**), `evaluation.json` |
| `--validate` | `validation.json` — p-values and verdicts |
| `--gate` | `cleared.json` — nodes passing both filters |
| `--status` | *(terminal)* the funnel per family |
| `--read <id>` | *(terminal)* the θ rows carrying a qualifying cell |

`--family <name>` restricts any build; `--rerun` rebuilds existing surfaces;
`--shifts N` sets resampling depth; `--require nominal` loosens the gate.

* * *

## Add your own asset

A workspace is one `universe.json`. No root code changes.

```json
{
  "meta": {
    "asset": {"provider": "yfinance", "ticker": "BTC-USD", "interval": "1d"},
    "start_date": "2015-01-01",
    "barrier_horizons": [3, 7, 14, 30],
    "theta":    {"min": -0.20, "max": 0.20, "step": 0.01},
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
  barrier.py         ← touch surfaces, quantile bins, economic filter, shuffle null
  writer.py          ← the xlsx deliverable
tree/tree.py         ← node traversal over universe.json
workspaces/<name>/
  universe.json      ← nodes + all asset config
  plugin.py          ← (optional) custom features and sources
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
- Politis & Romano (1994), *The Stationary Bootstrap*, JASA 89(428) — resampling that preserves serial dependence
- White (2000), *A Reality Check for Data Snooping*, Econometrica 68(5) — multiple testing over a searched universe of rules
- López de Prado (2018), *Advances in Financial Machine Learning*, ch. 3 — barrier labelling of financial time series
- Karatzas & Shreve (1991), *Brownian Motion and Stochastic Calculus* §3.5 — first-passage probabilities

**Indicators** — Wilder (1978) RSI/ATR · Appel (2005) MACD · Bollinger (2001) Bollinger Bands
