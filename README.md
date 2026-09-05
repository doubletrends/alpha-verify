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

The complete weighted-Bayes surface can be inverted into the form people actually use:
as of the configured historical demonstration date, how wide were the 25% and 50% touch
envelopes? All validation-cleared nodes contribute in their states on that date, and the
subsequent realized path is overlaid for an honest case study.

Every node is judged against three explicit gates: a raw node-wide null threshold,
false-discovery correction across the sweep, and practical effect size. The useful
result is the cleared set: conditions that pass all three.

## What It Found

| | `nasdaq_daily` | `btc_daily` |
|---|--:|--:|
| Nodes measured | 58 | 66 |
| Nodes shortlisted | 20 | 20 |
| Null tests (node x horizon) | 600 | 600 |
| Nodes passing raw null | 18 | 17 |
| Nodes passing BH/FDR | 19 | 17 |
| Economically usable | 19 | 20 |
| **Cleared all three** | **17** | **17** |

Three findings carry the front page.

**1. Barrier probability is widely predictable.** On NASDAQ daily, 17 of the 20
whole-node information leaders clear all three validation gates. Their strongest
discoveries are concentrated at short horizons: 16 at +1d and one at +2d.

**2. The filters are strict enough to be useful.** A condition has to pass raw
`node_peak_p ≤ 0.01`, BH/FDR `q ≤ 0.05`, and the economic threshold. The point is not a
large spreadsheet of indicators; it is a short list of claims that survived all three.

**3. Cross-asset regimes matter, but they are still magnitude regimes.** VIX, Treasury
yield and DXY features join price, volatility and trend features in the discovery set.
The report keeps the exact null diagnostics front and center, then renders one
conditional-shift surface for every node that cleared all three.

## Why The Numbers Are Trustworthy

### The Null Is Exact

Writing `counts[Δ, bin]` as a function of circular shift makes it a
cross-correlation between the touch indicator and the bin-membership mask. One FFT yields
every shift at once, returning the full permutation distribution rather than a resampled
approximation.

The finest obtainable p-value is `1/(n+1)`. For the NASDAQ daily run that floor is
`3.95e-4`; with 600 tests, Bonferroni at `alpha=0.05` would require `8.33e-5`, which is
below the floor. Validation therefore controls false discovery rate with
Benjamini-Hochberg instead of exposing an unusable family-wise threshold.

### Three Gates Are Required

Stage 3 only ranks information. Stage 4 requires `node_peak_p ≤ 0.01`, BH/FDR
`q ≤ 0.05`, and an economic deviation of at least 10 percentage points in the selected
representative bin, with at least 50 observations and at least two adjacent barrier rows
sharing the same sign. Only all three together constitute a cleared claim.

### The Baseline Is A Node

`baseline` is measured like any other node. Its single condition bin is the unconditional
touch probability surface, and every conditional node has to beat that reference. This
keeps the pipeline uniform and prevents special-case arithmetic from drifting away from
the artifacts.

## Do Conditions Compose?

Under conditional independence, log-odds contributions add, and every term already lives
in a cube cell:

```text
logit P(touch | x1..xk)
  = logit P(touch) + sum_i [logit P(touch | xi) - logit P(touch)]
```

Stage 6 tests that composition out of sample. Each training fold ranks all complete
ten-bin node tables, keeps its top 20, and cross-fits their log-odds contributions.
Non-negative ridge-logistic weights then reduce duplicated evidence without discarding
the residual information in overlapping nodes. All fitting happens before the test
window, with a 30-session embargo and non-overlapping scoring.

| model | Brier down | AUC | vs. prior |
|---|--:|--:|--:|
| constant prior | 0.214 | - | - |
| naive Bayes, fold-selected nodes | 0.279 | 0.614 | +30.3% worse |
| one node per family | 0.231 | 0.580 | +8.2% worse |
| + scale corrected | 0.194 | 0.561 | 9.2% better |
| weighted redundancy-aware | 0.254 | 0.616 | +18.9% worse |

The raw model is overconfident because correlated indicators count similar evidence many
times. For the NASDAQ headline target, `P(touch -7% within 30d)`, scale correction beats
the constant prior while learned node weights do not. On BTC, the weighted model does
best (Brier 0.197 versus 0.213 for the prior). The contrast is useful: retaining partial
information improves one workspace, but weighting is not automatically superior.

Across the complete NASDAQ composition sweep, 456 of 1,200 barrier/horizon targets beat the
prior after correction. The sweep includes rare-event targets, so its extreme AUC cells
are diagnostics rather than headline claims; inspect realized rates and scored counts in
`06_composition/composition.json` before interpreting any individual cell.

## Run It

```bash
pip install -e .

volatility-matrix surface          # 1. write 01_surface arrays and workbooks
volatility-matrix shift            # 2. write 02_shift arrays and workbooks
volatility-matrix selection        # 3. write 03_selection artifacts
volatility-matrix validation       # 4. exact nulls + BH/economic final verdicts
volatility-matrix redundancy       # 5. write redundancy NPZ, XLSX, and manifest
volatility-matrix composition      # 6. evaluate composition + write current probability sheet
volatility-matrix report           # 7. render workspaces/nasdaq_daily/07_report
volatility-matrix status           # inspect artifact and validation status
```

Every command defaults to `nasdaq_daily`. Pass `--workspace btc_daily` after the command
to run the BTC comparison workspace.

Pass a node ID to the read-only status command for a detailed shift and validation view:

```bash
volatility-matrix status vix_level --workspace nasdaq_daily
```

### What Lands On Disk

```text
workspaces/<name>/
  01_surface/<family>/<node>.npz          compute  full Δ x bins x horizons
  01_surface/<family>/<node>.xlsx         view     full cube workbook
  02_shift/<family>/<node>.npz            compute  full baseline-subtracted shift cube
  02_shift/<family>/<node>.xlsx           view     red/blue shift workbook
  03_selection/selection.json             compute  ranking index for selected nodes
  03_selection/<family>/rank_*.npz        compute  complete selected-node shift cubes
  03_selection/<family>/rank_*.xlsx       view     selected shift sheet workbook
  04_validation/<family>/<node>.npz       compute  exact shuffled null
  04_validation/<family>/<node>.xlsx      view     readable validation sheet
  04_validation/validation.json           three-gate verdicts and cleared nodes
  05_redundancy/redundancy.npz            numerical conditional-NMI matrix
  05_redundancy/redundancy.xlsx           readable redundancy workbook
  05_redundancy/redundancy.json           redundancy manifest and cluster summary
  06_composition/composition.npz           numerical predictions and surfaces
  06_composition/composition.json          composition metrics and metadata
  06_composition/probability.xlsx          current weighted probability surface
  06_composition/shift.xlsx                current weighted shift from baseline
  07_report/*.png                          report figures
```

Workspace contents generated by the pipeline, including all artifacts and report images,
are ignored by Git. Only each workspace's `universe.json` and optional `plugin.py` are
source-controlled.

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
