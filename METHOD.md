# Method

This repository measures conditional barrier-touch probabilities:

```text
P(price touches theta within h bars | condition bin)
```

That is different from asking where price closes. Stops, limits, liquidations and margin
calls care whether a level was reached at any point inside the forward window, so the
engine uses high/low extremes rather than close-to-close returns.

The product workflow has seven stages:

```text
01_surface_array/         --surface     compute: full theta x bins x horizons
01_surface_xlsx/          --surface     view: full surface workbooks

02_shift_array/           --shift       compute: full baseline-subtracted shift cube
02_shift_xlsx/            --shift       view: red/blue shift workbooks

03_selection_array/       --selection   compute: top skew-ranked node/bin sheets
03_selection_xlsx/        --selection   view: selected shift sheet workbooks

04_validation_array/      --validation  compute: exact shuffled nulls
04_validation_xlsx/       --validation  view: validation workbooks

05_gate.json              --gate        BH correction + economic intersect
06_bayes.npz              --bayes       walk-forward composition arrays
06_bayes.json             --bayes       walk-forward composition summary
workspaces/<name>/result/ --report      audience-facing figures
```

Stages 1-4 build the measured, selected and validated surfaces. Stage 5 applies the discovery
gate. Stage 6 is the out-of-sample composition check. Stage 7 renders figures from
stored artifacts and does not remeasure.

The dependency chain is one-way: shift reads surface artifacts, selection copies the
top-ranked sheet artifacts from shift, validation reads selection artifacts for the tested sheets, gate
reads validation artifacts, Bayes reads selected shift artifacts for its ordered feature
panel, and report reads the stored artifacts produced
by those stages. Validation is the one exception that may
recompute the circular-shift null from source data, because the null needs ordered
feature alignment rather than only aggregate cube rates.

## 0. Orient

```bash
python run.py --workspace <name> --status
```

The status table is the funnel: nodes declared, surfaces measured, shift artifacts
written, selected sheets copied, nulls validated, selected-sheet economics passed, and
sheets that cleared both the economic and statistical gates.

## 1. Surface

```bash
python run.py --workspace <name> --surface
python run.py --workspace <name> --surface --family volatility
```

This stage computes one probability cube per node:

| axis | contents |
|---|---|
| 0 | barrier level `theta`, ascending and including zero |
| 1 | condition bin, usually feature deciles |
| 2 | forward horizon `h` |

The value is the raw conditional probability:

```text
P(price touches theta within h | feature is in this bin)
```

Nothing is subtracted. The unconditional rate is the `baseline` node: a constant feature
whose single bin contains every bar. Measuring the baseline with the same engine keeps
the pipeline uniform and makes it the reference every conditional node is judged against.

Surface artifacts are written to:

```text
01_surface_array/<family>/<node>.npz
01_surface_xlsx/<family>/<node>.xlsx
```

The `.npz` file is the measurement record. The workbook is a readable projection: one
tab per condition bin, with barrier rows and horizon columns.

## 2. Shift

```bash
python run.py --workspace <name> --shift
python run.py --workspace <name> --shift --family volatility
```

The shift stage keeps the full Stage 1 grid and subtracts the baseline node from every
condition bin:

```text
shift = 100 * (P(price touches theta within h | bin) - P(price touches theta within h))
```

The shift stage does not judge nodes. It writes the full baseline-subtracted cube that
selection, validation and reporting read.

The economic filter is applied after sheet selection. A selected sheet passes the
economic filter when it has a cell that:

- deviates from the baseline by at least `min_dev` percentage points
- has at least `min_bin_n` observations in the bin
- persists across at least `min_run` adjacent theta rows with the same sign

The workbook uses a diverging scale centered at zero: red cells mean the barrier is
reached more often than baseline, blue cells mean less often. The adjacency requirement
is important. A single extreme cell can be a search artifact; a band of adjacent
barriers is closer to something a reader could actually use.

## 3. Selection

```bash
python run.py --workspace <name> --selection
python run.py --workspace <name> --selection --family volatility
```

Selection reads the Stage 2 shift cubes and treats every condition bin as its own
theta-by-horizon sheet. For each sheet it scores symmetric upside/downside skew:

```text
score = max |shift(+theta, h) - shift(-theta, h)|
```

Only positive `theta` rows are paired with their matching negative rows, and horizons
where the bin has fewer than `min_bin_n` observations are ignored. The stage ranks all
node/bin sheets globally, applies the economic filter to those sheets, and writes the
top `selection.top_k` rows, defaulting to 20, to
`03_selection_array/selection.json`, with one copied `.npz` per selected sheet under
`03_selection_array/<family>/` and one matching workbook under `03_selection_xlsx/<family>/`.

## 4. Validation

```bash
python run.py --workspace <name> --validation
python run.py --workspace <name> --validation --family volatility
```

Validation tests the selected node/bin sheets against an exact circular-shift null. The
feature is shifted against the price history, preserving its autocorrelation while
destroying its alignment with the future.

For each horizon, validation stores:

| field | shape | meaning |
|---|---:|---|
| `cell_real` | theta x bin x h | signed deviation from the node's own sample rate |
| `cell_p` | theta x bin x h | pointwise p-value for reading a surface |
| `cell_p95` | theta x bin x h | each cell's null 95th percentile |
| `sheet_peak_real` | bin x h | max absolute deviation over one selected sheet |
| `sheet_peak_p` | bin x h | p-value of that sheet search statistic |
| `sheet_peak_p95` | bin x h | null 95th percentile of that sheet statistic |
| `peak_real` | h | max absolute deviation over the surface |
| `peak_p` | h | p-value of that max search statistic |
| `peak_p95` | h | null 95th percentile of that max statistic |
| `n_shifts` | h | usable shifts, giving the p-value floor `1/(n+1)` |

The exact null is fast because counts by circular shift are a cross-correlation. One FFT
returns every shift. The floor matters: with only `n` distinct shifts, no method can
honestly report a p-value below `1/(n+1)`.

`cell_p` is pointwise. It helps read where a surface is unusual, but it is not a
discovery criterion. The gate uses `sheet_peak_p` for the sheets selected by
`03_selection_array/selection.json`, which accounts for the search over each 2D sheet.

## 5. Gate

```bash
python run.py --workspace <name> --gate
python run.py --workspace <name> --gate --fdr 0.05
```

The gate applies Benjamini-Hochberg correction across the selected sheet/horizon sweep,
then intersects those discoveries with the economic filter stored in
`03_selection_array/selection.json`.

A selected sheet clears the product claim only when both are true:

- **Economic:** a stable, useful-size deviation from the baseline exists.
- **Statistical:** the sheet's peak survives the exact shuffled-null test after FDR
  correction.

The result is written to `05_gate.json`:

```text
summary.discovery   selected sheets/horizons that survive the null
summary.nominal     pointwise p <= 0.05 before FDR
summary.cleared     selected sheets that cleared both filters
cleared[]           the headline rows
tests[]             every validation test with verdict and q-value
```

Bonferroni is intentionally not used. With hundreds of tests, its threshold falls below
the exact null's p-value floor, making it unreachable on these workspaces. BH controls
the expected false-discovery share among discoveries, which is the useful correction for
a screen of this size.

## 6. Compose

```bash
python run.py --workspace <name> --bayes
```

Stages 1-5 evaluate one selected condition sheet at a time. Composition asks whether the
selected conditions can be combined out of sample.

Under conditional independence, log-odds add:

```text
logit P(touch theta in h | x1..xk)
  = logit P(touch theta in h)
    + sum_i [logit P(touch theta in h | xi) - logit P(touch theta in h)]
```

Every term already lives in the measured cubes. The composition stage tests this with an
expanding walk-forward design:

1. Fit bin edges, per-bin rates and prior on the training window only.
2. Embargo the last `horizon` training bars so labels do not reach into test data.
3. Keep one feature per family based only on training-window separation.
4. Predict the test block.
5. Score every `horizon`th bar so forward windows do not overlap.

It writes:

```text
06_bayes.npz
06_bayes.json
```

The report compares prior-only, all-node naive Bayes, one-node-per-family, and Platt
scale-corrected probabilities. The point is not to claim a trading strategy; it is to
test whether the measured conditional tables compose without leaking future data.

## 7. Report

```bash
python run.py --workspace <name> --report
```

The report renders `workspaces/<name>/result/*.png` plus a local index. Figures read
from `.npz` and `.json` artifacts; they do not recompute the pipeline.

The main product figures show:

- the measured forward envelope from the latest close
- the exact null distribution behind a headline node
- one conditional-shift surface for every node that cleared both
- the composition ranking grid
- the ATR regime ladder
- the null-gap ranking across top discoveries

## Adding a Feature

Register reusable feature functions in `data/features.py`:

```python
def _my_feature(close: pd.Series, period: int) -> pd.Series:
    ...

register("my_feature", lambda d, p: _my_feature(d["close"], p["period"]))
```

Workspace-specific sources and features live in `workspaces/<name>/plugin.py`, which is
loaded automatically. Then add a node to `workspaces/<name>/universe.json`.

There is no progress field. Progress is inferred from artifacts on disk.

## Adding a Workspace

`universe.json` `meta` carries the asset-specific configuration:

| field | meaning |
|---|---|
| `asset` | `{provider, ticker, interval}` |
| `start_date` | history start |
| `horizons` | full horizon ladder |
| `theta` | full barrier ladder |
| `n_bins` | condition bins per feature |
| `evaluate` | economic filter thresholds |
| `bayes` | composition target and fold count |

Scale `theta` to the asset and horizon. BTC daily can support wider barriers than a
daily equity index; a good workspace spends barrier rows where touches occur often
enough to estimate.

## Files

| path | written by | contents |
|---|---|---|
| `universe.json` | you | nodes and asset config |
| `01_surface_array/<family>/<node>.npz` | `--surface` | full measured cube |
| `01_surface_xlsx/<family>/<node>.xlsx` | `--surface` | readable full workbook |
| `02_shift_array/<family>/<node>.npz` | `--shift` | full baseline-subtracted shift cube |
| `02_shift_xlsx/<family>/<node>.xlsx` | `--shift` | red/blue shift workbook |
| `03_selection_array/selection.json` | `--selection` | ranking index for selected sheets |
| `03_selection_array/<family>/rank_*.npz` | `--selection` | copied selected shift sheet arrays |
| `03_selection_xlsx/<family>/rank_*.xlsx` | `--selection` | readable selected shift sheet workbooks |
| `04_validation_array/<family>/<node>.npz` | `--validation` | exact null artifacts |
| `04_validation_xlsx/<family>/<node>.xlsx` | `--validation` | readable validation workbook |
| `05_gate.json` | `--gate` | FDR verdicts and cleared-both rows |
| `06_bayes.npz` | `--bayes` | pooled walk-forward predictions |
| `06_bayes.json` | `--bayes` | walk-forward metrics and fold metadata |
| `workspaces/<name>/result/*.png` | `--report` | product figures |

Rendered workbooks are git-ignored. The `.npz` files are the measurement record.
