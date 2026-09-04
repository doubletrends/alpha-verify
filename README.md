![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-2.0-013243?logo=numpy&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-2.2-150458?logo=pandas&logoColor=white)
![matplotlib](https://img.shields.io/badge/matplotlib-3.8-11557c)
![openpyxl](https://img.shields.io/badge/openpyxl-3.1-1D6F42)

# Conditional Barrier Probabilities

**Not "where will price close later?" - "did price ever reach this level?"** Stops,
limits, margin calls and liquidations all turn on a barrier touch, not a closing print.
This pipeline measures that question across barrier levels, condition bins and forward
horizons, using high/low extremes and an exact shuffled null.

The flagship workspace is now `nasdaq_daily`: a reproducible daily NASDAQ
Composite run from 2015-01-01, with a 30-session horizon ladder and a barrier grid scaled
for equity-index moves.

![conditional band](workspaces/nasdaq_daily/result/01_band.png)

The figure above inverts the measured cube into the form people actually use: given the
condition the index is in on the latest bar, how wide is the historical 50% touch
envelope? The filled band is the current condition bin; the thin bands show the edge
bins, so the value of conditioning is visible on the price axis.

Every node is judged against the two filters that decide whether it clears: practical
effect size and shuffled-null strength. The useful result is the cleared-both set:
conditions that are large enough to matter and too strong to explain as alignment noise.

## What It Found

| | `nasdaq_daily` | `btc_daily` |
|---|--:|--:|
| Nodes measured | 58 | 66 |
| Null tests (node x horizon) | 406 | 462 |
| Economically usable | 53 | 55 |
| Nodes with a BH discovery | 53 | 51 |
| **Cleared both** | **51** | **50** |

Three findings carry the front page.

**1. Barrier probability is widely predictable.** On NASDAQ daily, 51 of 58 nodes clear
both the economic filter and the shuffled-null gate. The strongest cleared cells are not
all one-bar artifacts: the peak horizons are +2d for 13 nodes, +7d for 12, +1d for 10,
and +30d for 7.

**2. The filters are strict enough to be useful.** A condition has to pass an economic
threshold and a shuffled-null test corrected across the sweep. The point is not a large
spreadsheet of indicators; it is a short list of conditional barrier claims that survived
both gates.

**3. Cross-asset regimes matter, but they are still magnitude regimes.** VIX, Treasury
yield and DXY features join price, volatility and trend features in the discovery set.
The report keeps the exact null diagnostics front and center, then renders one
conditional-shift surface for every node that cleared both.

## Why The Numbers Are Trustworthy

### The Null Is Exact

Writing `counts[theta, bin]` as a function of circular shift makes it a
cross-correlation between the touch indicator and the bin-membership mask. One FFT yields
every shift at once, returning the full permutation distribution rather than a resampled
approximation.

![the null](workspaces/nasdaq_daily/result/05_null.png)

The finest obtainable p-value is `1/(n+1)`. For the NASDAQ daily run that floor is
`3.95e-4`; with 406 tests, Bonferroni at `alpha=0.05` would require `1.23e-4`, which is
below the floor. The gate therefore controls false discovery rate with
Benjamini-Hochberg instead of exposing an unusable family-wise threshold.

### Two Filters Are Required

The economic filter requires a deviation of at least 10 percentage points, in a bin with
at least 50 observations, across at least two adjacent barrier rows with the same sign.
The statistical filter requires the surface peak to clear the shuffled null after
correction across the sweep. Neither filter alone is treated as a claim.

### The Baseline Is A Node

`baseline` is measured like any other node. Its single condition bin is the unconditional
touch probability surface, and every conditional node has to beat that reference. This
keeps the pipeline uniform and prevents special-case arithmetic from drifting away from
the artifacts.

![conditional shift](workspaces/nasdaq_daily/result/08_conditional_shift_vix_level.png)

### Skew Is Informative

The mirrored summary grid also produces a skew artifact:
`P(touch +theta | bin) - P(touch -theta | bin)`, with baseline skew subtracted in the
excess view. Skew is part of the product because it makes up/down asymmetry easy to
inspect, but it is not validated and does not feed cleared-both.

## Do Conditions Compose?

Under conditional independence, log-odds contributions add, and every term already lives
in a cube cell:

```text
logit P(touch | x1..xk)
  = logit P(touch) + sum_i [logit P(touch | xi) - logit P(touch)]
```

Stage 6 tests that composition out of sample. Bin edges, per-bin rates, the prior and
the scale correction are all fit on an expanding training window, then applied to later
bars with a 30-session embargo and non-overlapping scoring.

| model | Brier down | AUC | vs. prior |
|---|--:|--:|--:|
| constant prior | 0.206 | - | - |
| naive Bayes, all 57 nodes | 0.333 | 0.576 | +61.4% worse |
| one node per family | 0.233 | 0.580 | +13.2% worse |
| **+ scale corrected** | **0.197** | 0.534 | **-4.4%** |

The raw model is overconfident because correlated indicators count similar evidence many
times. Keeping one node per family removes most of the damage; a Platt scale correction
does the rest for the headline target, `P(touch -7% within 30d)`.

![composition grid](workspaces/nasdaq_daily/result/10_composition_grid.png)

Across the judged grid, 73 of 126 usable targets beat the prior out of sample. The
strongest ranked target is `+10%` within `+3d`, at AUC 0.99.

## Run It

```bash
pip install -e .

python run.py --surface            # 1. write 01_surface_array and 01_surface_xlsx
python run.py --summary            # 2. write 02_summary_array and 02_summary_xlsx
python run.py --validation         # 3. write 03_validation_array and 03_validation_xlsx
python run.py --skew               # 4. write 04_skew_array and 04_skew_xlsx
python run.py --gate               # 5. BH correction and economic/statistical intersect
python run.py --bayes              # 6. walk-forward composition
python run.py --report             # 7. render workspaces/nasdaq_daily/result

python run.py --status
python run.py --read vix_level
```

`run.py` defaults to `nasdaq_daily`. After `pip install -e .`, the same CLI is
also available as `volatility-matrix`. Pass `--workspace btc_daily` to rerun the
BTC comparison workspace. `--family <name>` restricts a stage, `--rerun` rebuilds
existing artifacts, and `--fdr Q` sets the Benjamini-Hochberg rate used by `--gate`.

### What Lands On Disk

```text
workspaces/<name>/
  01_surface_array/<family>/<node>.npz              compute  full theta x bins x horizons
  01_surface_xlsx/<family>/<node>.xlsx          view     full cube workbook
  02_summary_array/<family>/<node>.npz      compute  judged theta x bins x horizons
  02_summary_xlsx/<family>/<node>.xlsx  view     summary cube workbook
  03_validation_array/<family>/<node>.npz         compute  exact shuffled null
  03_validation_xlsx/<family>/<node>.xlsx   view     readable validation sheet
  04_skew_array/<family>/<node>.npz               compute  P(+theta) vs P(-theta)
  04_skew_xlsx/<family>/<node>.xlsx         view     readable skew sheet
  evaluation.json  05_gate.json  06_bayes.npz  06_bayes.json
                                                   verdicts and composition metrics
workspaces/<name>/result/*.png                                report figures
```

The `.npz` arrays are tracked because they are the measurement record. Rendered `.xlsx`
workbooks are ignored: they are regenerated by `--surface`, `--summary`, `--validation`,
and `--skew`.

## Limits

**Sample size.** A decile condition on ten years of daily bars has only a few hundred
bars, and far fewer non-overlapping windows at long horizons. Counts are bar counts, not
independent-trial counts.

**Bar resolution.** Daily OHLC records the session high and low, not which came first.
Questions that depend on the order of a stop and take-profit inside the same daily bar
are not answerable from daily data.

**Hourly reproducibility.** The old `nasdaq_hourly_24hrs` workspace was removed because
Yahoo only serves a trailing hourly window. The daily NASDAQ workspace is reproducible
across time in a way the hourly one was not.

## References

**Data** - [`yfinance`](https://github.com/ranaroussi/yfinance) and
[CoinMetrics Community API](https://docs.coinmetrics.io/api/v4)

**Method** - Davison & Hinkley (1997), Benjamini & Hochberg (1995), Platt (1999),
White (2000), Lopez de Prado (2018), Politis & Romano (1994), and Karatzas & Shreve
(1991). Indicator definitions follow Wilder RSI/ATR, Appel MACD and Bollinger Bands.

See [METHOD.md](METHOD.md) for the artifact formats, stage-by-stage procedure
and extension points.
